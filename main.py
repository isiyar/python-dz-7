import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import aiohttp
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor


def load_city(df):
    df["rolling_mean"] = df["temperature"].rolling(window=30, min_periods=1).mean()

    season_stat = df.groupby("season")["temperature"].agg(["mean", "std"]).reset_index()
    season_stat.rename(
        columns={"mean": "season_mean", "std": "season_std"}, inplace=True
    )

    df = df.merge(season_stat, on="season", how="left")

    df["is_anomaly"] = (
        df["temperature"] > df["rolling_mean"] + 2 * df["season_std"]
    ) | (df["temperature"] < df["rolling_mean"] - 2 * df["season_std"])

    return df


def load_data_seq(df):
    results = []
    for city, group in df.groupby("city"):
        results.append(load_city(group.copy()))
    return pd.concat(results)


def load_data_par(df):
    city_groups = [group.copy() for _, group in df.groupby("city")]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(load_city, city_groups))
    return pd.concat(results)


def get_weather_sync(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    return requests.get(url)


async def fetch_weather_async(session, city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with session.get(url) as response:
        return await response.json(), response.status


async def get_all_weather_async(cities, api_key):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_weather_async(session, city, api_key) for city in cities]
        return await asyncio.gather(*tasks)


st.set_page_config(page_title="Анализ температур", layout="wide")
st.title("Анализ температур")

st.sidebar.header("Настройки")
history_file = st.sidebar.file_uploader(
    "Загрузите файл с историческими данными", type=["csv"]
)

if history_file is not None:
    df = pd.read_csv(history_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    cities = df["city"].unique().tolist()
    selected_city = st.sidebar.selectbox("Выберите город", cities)
    api_key = st.sidebar.text_input("Введите API-ключ", type="password")

    st.markdown("---")

    st.subheader("Параллельный и последовательный анализ")
    if st.button("Сравнить"):
        start_seq = time.time()
        data_seq = load_data_seq(df)
        time_seq = time.time() - start_seq

        start_par = time.time()
        data_par = load_data_par(df)
        time_par = time.time() - start_par

        st.write(f"**Последовательное выполнение:** {time_seq:.4f} секунд")
        st.write(f"**Параллельное выполнение:** {time_par:.4f} секунд")
        st.info(
            "Вывод: для операций с небольшим кол-вом данных подойдет последовательное выполнения, для больших объемов лучше использовать параллельное"
        )

    proc_df = load_data_seq(df)
    city_data = proc_df[proc_df["city"] == selected_city].sort_values("timestamp")

    st.header(f"Анализ исторических данных: {selected_city}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Средняя температура", f"{city_data['temperature'].mean():.1f} °C")
    col2.metric("Максимальная температура", f"{city_data['temperature'].max():.1f} °C")
    col3.metric("Минимальная температура", f"{city_data['temperature'].min():.1f} °C")

    fig1 = px.line(
        city_data,
        title="Температура и скользящее среднее",
    )
    fig1.add_trace(
        go.Scatter(
            x=city_data["timestamp"],
            y=city_data["rolling_mean"],
            mode="lines",
            name="Скользящее среднее",
            line=dict(color="orange"),
        )
    )

    anomalies = city_data[city_data["is_anomaly"]]
    fig1.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["temperature"],
            mode="markers",
            name="Аномалии",
            marker=dict(color="red", size=6),
        )
    )
    st.plotly_chart(fig1, use_container_width=True)

    season_stat = (
        city_data.groupby("season")["temperature"].agg(["mean", "std"]).reset_index()
    )
    fig2 = px.bar(
        season_stat,
        x="season",
        y="mean",
        error_y="std",
        title="Средняя температура по сезонам",
        labels={"mean": "Средняя температура", "season": "Сезон"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.header("Текущая погода")

    if api_key:
        if st.button("Сравнить скорость синхронных и асинхронных запросов"):
            st.write("Загрузка")

            start_sync = time.time()
            for c in cities:
                get_weather_sync(c, api_key)
            sync_time = time.time() - start_sync

            start_async = time.time()
            asyncio.run(get_all_weather_async(cities, api_key))
            async_time = time.time() - start_async

            st.write(f"**Синхронные запросы:** {sync_time:.4f} секунд")
            st.write(f"**Асинхронные запросы:** {async_time:.4f} секунд")
            st.success(
                "Вывод: асинхронные запросы гораздо быстрее, т.к. запросы выполняются одновременно, не дожидаясь выполнения предыдущих"
            )

        response = get_weather_sync(selected_city, api_key)

        if response.status_code == 200:
            weather_data = response.json()
            current_temp = weather_data["main"]["temp"]

            current_month = pd.Timestamp.now().month
            if current_month in [12, 1, 2]:
                current_season = "winter"
            elif current_month in [3, 4, 5]:
                current_season = "spring"
            elif current_month in [6, 7, 8]:
                current_season = "summer"
            else:
                current_season = "autumn"

            season_data = season_stat[season_stat["season"] == current_season].iloc[0]
            mean_temp = season_data["mean"]
            std_temp = season_data["std"]

            is_normal = (
                (mean_temp - 2 * std_temp) <= current_temp <= (mean_temp + 2 * std_temp)
            )

            st.metric(f"Текущая температура в {selected_city}", f"{current_temp} °C")

            if is_normal:
                st.success(f"Температура в пределах сезонной нормы")
            else:
                st.error(f"Аномалия, температура выходит за пределы сезонной нормы")

        elif response.status_code == 401:
            st.error(
                '{"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}'
            )
        else:
            st.error(f"Ошибка: {response.status_code}")

    else:
        st.warning("Введите API-ключ для просмотра текущей погоды")

else:
    st.info("Загрузите файл с историческими данными")
