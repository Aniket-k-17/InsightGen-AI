# core/visualization.py
# This file creates charts using Plotly.
# Plotly makes interactive charts - you can hover, zoom, and click on them.
# Each function takes a DataFrame and returns a Plotly figure (chart object).

import plotly.express as px  # plotly.express is the simple, easy way to make charts


def make_histogram(df, column):
    """
    A histogram shows how values are DISTRIBUTED in a column.
    Example: how many people earn 0-10k, 10k-20k, 20k-30k etc.
    """
    chart = px.histogram(
        df,                          # the data table
        x=column,                   # which column goes on the x-axis
        nbins=40,                   # how many bars to show
        title=f"Distribution of {column}",
        color_discrete_sequence=["#4f8ef7"],  # bar colour (blue)
        template="plotly_dark",     # dark background theme
    )
    return chart


def make_scatter_plot(df, x_column, y_column):
    """
    A scatter plot shows the RELATIONSHIP between two number columns.
    Each dot = one row of data. If dots form a line, the columns are related.
    """
    chart = px.scatter(
        df,
        x=x_column,   # column on horizontal axis
        y=y_column,   # column on vertical axis
        title=f"{x_column} vs {y_column}",
        opacity=0.6,  # make dots slightly see-through so overlapping ones are visible
        template="plotly_dark",
    )
    return chart


def make_bar_chart(df, column, top_n=10):
    """
    A bar chart shows how often each VALUE appears in a TEXT column.
    Example: how many sales per country.
    top_n = only show the top N most common values.
    """
    # value_counts() counts how many times each value appears
    # head(top_n) keeps only the top N rows
    counts = df[column].value_counts().head(top_n).reset_index()
    counts.columns = ["Value", "Count"]  # rename columns for the chart

    chart = px.bar(
        counts,
        x="Count",
        y="Value",
        orientation="h",  # horizontal bars (easier to read long category names)
        title=f"Top {top_n} values in '{column}'",
        color="Count",
        color_continuous_scale="Blues",
        template="plotly_dark",
    )
    return chart


def make_box_plot(df, columns):
    """
    A box plot shows the SPREAD of data and highlights outliers.
    The box shows where 50% of data falls. Dots outside are outliers.
    columns can be a single column name (string) or a list of column names.
    """
    # If an empty list was passed in, fall back to all numeric columns
    if isinstance(columns, list) and len(columns) == 0:
        columns = df.select_dtypes(include="number").columns.tolist()

    # If still nothing to plot, return a blank figure with a message
    if not columns:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title="Box Plot — No numeric columns available",
            template="plotly_dark",
        )
        return fig

    chart = px.box(
        df,
        y=columns,       # which columns to show (can be multiple)
        title="Box Plot — Spread and Outliers",
        template="plotly_dark",
    )
    return chart


def make_violin_plot(df, column):
    """
    A violin plot is like a box plot but shows the full SHAPE of the distribution.
    Wide = more data at that value. Narrow = fewer data points there.
    """
    chart = px.violin(
        df,
        y=column,
        box=True,           # show a small box plot inside the violin
        points="outliers",  # show individual outlier dots
        title=f"Violin Plot of {column}",
        color_discrete_sequence=["#7c5cfc"],
        template="plotly_dark",
    )
    return chart


def make_correlation_heatmap(df):
    """
    A heatmap shows CORRELATION between all number columns.
    Correlation = how much two columns move together.
    +1 = perfectly move together. -1 = perfectly opposite. 0 = no relationship.
    """
    # Select only number columns
    number_columns = df.select_dtypes(include="number").columns

    # .corr() calculates correlation between every pair of number columns
    correlation_matrix = df[number_columns].corr().round(2)

    chart = px.imshow(
        correlation_matrix,
        text_auto=True,       # show the number values inside each cell
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",  # red=negative, blue=positive
        zmin=-1, zmax=1,      # range of correlation values
        template="plotly_dark",
    )
    return chart


def make_pie_chart(df, column, top_n=8):
    """
    A pie chart shows what SHARE each category takes of the whole.
    """
    counts = df[column].value_counts().head(top_n)

    chart = px.pie(
        names=counts.index,    # category names (become slice labels)
        values=counts.values,  # the size of each slice
        title=f"Share of values in '{column}'",
        hole=0.4,              # 0.4 makes it a donut chart (looks cleaner)
        template="plotly_dark",
    )
    return chart


def get_auto_charts(df):
    """
    Automatically creates the best set of charts for a given DataFrame.
    Returns a dictionary: {"Chart Name": chart_object, ...}
    The Upload and Dashboard pages use this function.
    """
    charts = {}  # empty dictionary we will fill with charts

    number_columns = df.select_dtypes(include="number").columns.tolist()
    text_columns   = df.select_dtypes(include="object").columns.tolist()

    # If there is at least 1 number column, make these charts
    if len(number_columns) >= 1:
        charts["Histogram"]  = make_histogram(df, number_columns[0])
        charts["Violin"]     = make_violin_plot(df, number_columns[0])
        charts["Box Plot"]   = make_box_plot(df, number_columns[:6])  # max 6 columns

    # If there are at least 2 number columns, make scatter + heatmap
    if len(number_columns) >= 2:
        charts["Scatter"]    = make_scatter_plot(df, number_columns[0], number_columns[1])
        charts["Heatmap"]    = make_correlation_heatmap(df)

    # If there is at least 1 text column, make bar + pie
    if len(text_columns) >= 1:
        charts["Bar Chart"]  = make_bar_chart(df, text_columns[0])
        charts["Pie Chart"]  = make_pie_chart(df, text_columns[0])

    return charts