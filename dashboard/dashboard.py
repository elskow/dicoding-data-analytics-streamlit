import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.sidebar.title("E-commerce Order Performance and Customer Satisfaction Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("No file uploaded. Using pre-loaded data.")
    df = pd.read_csv("dashboard/ecommerce.csv")

page = st.sidebar.selectbox(
    "Drilldown",
    [
        "Overview",
        "Shipping Delays and Customer Satisfaction",
        "Delayed Deliveries by Region",
        "Customer Segmentation",
    ],
)

if page == "Overview":
    unique_customers = df["customer_unique_id"].nunique()
    avg_review_score = df["review_score"].mean()
    top_product_category = df["product_category_name_english"].mode()[0]
    top_product_category_count = (
        df["product_category_name_english"].value_counts().max()
    )
    total_revenue = df["payment_value"].sum()
    avg_order_value = df["payment_value"].mean()

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M")
    monthly_revenue = df.groupby("month")["payment_value"].sum()
    monthly_revenue_growth = monthly_revenue.pct_change()
    latest_month_growth = monthly_revenue_growth.iloc[-1] * 100

    st.subheader("Key Metrics")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric(label="Unique Customers", value=unique_customers)
    col2.metric(label="Average Review Score", value=f"{avg_review_score:.2f}")
    col3.metric(
        label="Top Product Category",
        value=top_product_category,
        delta=str(top_product_category_count),
    )
    col4.metric(label="Total Revenue", value=f"${total_revenue:,.2f}")
    col5.metric(label="Average Order Value", value=f"${avg_order_value:,.2f}")
    col6.metric(label="Monthly Revenue Growth", value=f"{latest_month_growth:.2f}%")

    st.subheader("Monthly Total Revenue")
    fig = px.line(
        monthly_revenue,
        x=monthly_revenue.index.astype(str),
        y=monthly_revenue.values,
        labels={"x": "Month", "y": "Total Revenue ($)"},
    )
    st.plotly_chart(fig)

elif page == "Shipping Delays and Customer Satisfaction":
    data = df[
        [
            "order_delivered_customer_date",
            "order_delivered_carrier_date",
            "review_score",
        ]
    ]
    data["order_delivered_customer_date"] = pd.to_datetime(
        data["order_delivered_customer_date"]
    )
    data["order_delivered_carrier_date"] = pd.to_datetime(
        data["order_delivered_carrier_date"]
    )
    data["delay_days"] = (
        data["order_delivered_customer_date"] - data["order_delivered_carrier_date"]
    ).dt.days

    data["delay_days"] = data["delay_days"].apply(lambda x: max(x, 0))

    bins = [-1, 0, 3, 7, 14, float("inf")]
    labels = ["No Delay", "1-3 Days", "4-7 Days", "8-14 Days", "15+ Days"]
    data["delay_category"] = pd.cut(
        data["delay_days"], bins=bins, labels=labels, ordered=True
    )

    with st.sidebar:
        st.header("Filters")
        delay_category = st.multiselect(
            "Delay Categories", options=labels, default=labels
        )
        date_range = st.slider(
            "Delivery Date Range",
            min_value=data["order_delivered_customer_date"].min().date(),
            max_value=data["order_delivered_customer_date"].max().date(),
            value=(
                data["order_delivered_customer_date"].min().date(),
                data["order_delivered_customer_date"].max().date(),
            ),
        )
        delay_days = st.slider(
            "Delay Days Range",
            min_value=0,  # Start from 0
            max_value=int(data["delay_days"].max()),
            value=(0, int(data["delay_days"].max())),
        )

    filtered_data = data[data["delay_category"].isin(delay_category)]
    filtered_data = filtered_data[
        (
            filtered_data["order_delivered_customer_date"]
            >= pd.to_datetime(date_range[0])
        )
        & (
            filtered_data["order_delivered_customer_date"]
            <= pd.to_datetime(date_range[1])
        )
    ]

    st.subheader("Shipping Delay vs Review Score")
    fig = px.box(
        filtered_data,
        x="delay_category",
        y="review_score",
        color="delay_category",
        category_orders={"delay_category": labels},
        labels={
            "delay_category": "Shipping Delay Category",
            "review_score": "Review Score",
        },
    )
    st.plotly_chart(fig)

    filtered_delay_data = data[
        (data["delay_days"] >= delay_days[0]) & (data["delay_days"] <= delay_days[1])
    ]
    st.subheader("Distribution of Delay Days")
    fig = px.histogram(filtered_delay_data, x="delay_days", nbins=30)
    st.plotly_chart(fig)

elif page == "Delayed Deliveries by Region":
    data = df[
        [
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "customer_state",
            "geolocation_lng",
            "geolocation_lat",
        ]
    ]
    data["order_delivered_customer_date"] = pd.to_datetime(
        data["order_delivered_customer_date"]
    )
    data["order_estimated_delivery_date"] = pd.to_datetime(
        data["order_estimated_delivery_date"]
    )
    data["delayed"] = (
        data["order_delivered_customer_date"] > data["order_estimated_delivery_date"]
    )

    delay_by_region = data.groupby("customer_state")["delayed"].mean().reset_index()
    delay_by_region["delayed"] *= 100

    with st.sidebar:
        st.header("Filters")
        regions = st.multiselect(
            "Regions",
            options=delay_by_region["customer_state"].unique(),
            default=delay_by_region["customer_state"].unique(),
        )

    filtered_region_data = delay_by_region[
        delay_by_region["customer_state"].isin(regions)
    ].sort_values("delayed", ascending=False)

    st.subheader("Delayed Deliveries by Region")
    fig = px.bar(
        filtered_region_data,
        x="customer_state",
        y="delayed",
        color="delayed",
        labels={"customer_state": "Region", "delayed": "Delayed Deliveries (%)"},
    )
    st.plotly_chart(fig)

    geo_data = data[data["customer_state"].isin(regions)]
    st.subheader("Geographical Distribution of Delayed Deliveries")
    fig = px.scatter_geo(
        geo_data, lat="geolocation_lat", lon="geolocation_lng", color="delayed"
    )
    st.plotly_chart(fig)

    delay_proportion = data["delayed"].value_counts(normalize=True).reset_index()
    delay_proportion.columns = ["Delayed", "Proportion"]
    st.subheader("Proportion of Delayed Deliveries")
    fig = px.pie(delay_proportion, names="Delayed", values="Proportion")
    st.plotly_chart(fig)

elif page == "Customer Segmentation":
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            {
                "order_purchase_timestamp": lambda x: (
                    df["order_purchase_timestamp"].max() - x.max()
                ).days,
                "customer_unique_id": "count",
                "payment_value": "sum",
            }
        )
        .rename(
            columns={
                "order_purchase_timestamp": "recency",
                "customer_unique_id": "frequency",
                "payment_value": "monetary",
            }
        )
    )

    recency_threshold = rfm["recency"].median()
    frequency_threshold = rfm["frequency"].median()
    monetary_threshold = rfm["monetary"].median()

    def assign_cluster(row):
        if (
            row["recency"] <= recency_threshold
            and row["frequency"] > frequency_threshold
            and row["monetary"] > monetary_threshold
        ):
            return "Best Customers"
        elif (
            row["recency"] <= recency_threshold
            and row["frequency"] > frequency_threshold
        ):
            return "Loyal Customers"
        elif row["recency"] <= recency_threshold:
            return "Recent Customers"
        else:
            return "At Risk"

    rfm["cluster"] = rfm.apply(assign_cluster, axis=1)

    with st.sidebar:
        st.header("Filters")
        clusters = st.multiselect(
            "Customer Segments",
            options=rfm["cluster"].unique(),
            default=rfm["cluster"].unique(),
        )

    filtered_rfm = rfm[rfm["cluster"].isin(clusters)]

    st.subheader("Customer Segmentation")
    cluster_counts = filtered_rfm["cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Customer Segment", "Number of Customers"]

    fig = px.bar(
        cluster_counts,
        x="Customer Segment",
        y="Number of Customers",
        color="Customer Segment",
    )
    st.plotly_chart(fig)

    st.subheader("Customer Data")
    st.dataframe(filtered_rfm)

    st.subheader("Recency vs Frequency")
    fig = px.scatter(
        filtered_rfm, x="recency", y="frequency", size="monetary", color="cluster"
    )
    st.plotly_chart(fig)

    st.subheader("Scatter Matrix of RFM")
    fig = px.scatter_matrix(
        filtered_rfm, dimensions=["recency", "frequency", "monetary"], color="cluster"
    )
    st.plotly_chart(fig)
