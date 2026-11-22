
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset path
DATASET_PATH = BASE_DIR / "dataset" / "cleaned_dataset.csv"

# Output directory for graphs
OUTPUT_DIR = BASE_DIR / "static" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------
# 1. LOAD CLEANED DATASET
# --------------------------------

def load_data():

    df = pd.read_csv(DATASET_PATH)

    print("Cleaned dataset loaded successfully!")

    return df


# --------------------------------
# 2. DESCRIPTIVE STATISTICS
# --------------------------------

def descriptive_statistics(df):

    print("\n========== DESCRIPTIVE STATISTICS ==========")

    features = df.drop(
        columns=["subject", "sessionIndex", "rep"]
    )

    print(features.describe())

    features.describe().to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv"
    )


# --------------------------------
# 3. H.PERIOD HISTOGRAM
# --------------------------------

def plot_histogram(df):

    print("\nGenerating H.period histogram...")

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="H.period",
        bins=30,
        kde=True
    )

    plt.title("Distribution of H.period")
    plt.xlabel("Hold Time (seconds)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "h_period_histogram.png",
        dpi=300
    )

    plt.close()

    print("Histogram saved successfully!")


# --------------------------------
# 4. CORRELATION HEATMAP
# --------------------------------

def plot_correlation(df):

    print("\nGenerating correlation heatmap...")

    features = df.drop(
        columns=["subject", "sessionIndex", "rep"]
    )

    correlation = features.corr()

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=False
    )

    plt.title("Correlation Between Timing Features")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()

    print("Correlation heatmap saved successfully!")


# --------------------------------
# 5. SUBJECT-WISE AVERAGES
# --------------------------------

def subject_analysis(df):

    print("\nGenerating subject-wise analysis...")

    subject_means = df.groupby("subject")[
        "H.period"
    ].mean()

    print("\nAverage H.period for each subject:")
    print(subject_means)

    subject_means.to_csv(
        OUTPUT_DIR / "subject_averages.csv"
    )

    plt.figure(figsize=(16, 6))

    subject_means.plot(kind="bar")

    plt.title("Average H.period by Subject")
    plt.xlabel("Subject")
    plt.ylabel("Average Hold Time (seconds)")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "subject_averages.png",
        dpi=300
    )

    plt.close()

    print("Subject analysis saved successfully!")


# --------------------------------
# 6. SUBJECT-WISE BOXPLOT
# --------------------------------

def plot_subject_boxplot(df):

    print("\nGenerating subject-wise boxplot...")

    plt.figure(figsize=(18, 7))

    sns.boxplot(
        data=df,
        x="subject",
        y="H.period"
    )

    plt.title("H.period Distribution Across Subjects")
    plt.xlabel("Subject")
    plt.ylabel("Hold Time (seconds)")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "subject_boxplot.png",
        dpi=300
    )

    plt.close()

    print("Subject boxplot saved successfully!")


# --------------------------------
# 7. MAIN EXECUTION
# --------------------------------

if __name__ == "__main__":

    print("\n========== KEYGUARD EDA ==========")

    data = load_data()

    descriptive_statistics(data)

    plot_histogram(data)

    plot_correlation(data)

    subject_analysis(data)

    plot_subject_boxplot(data)

    print("\nAll EDA operations completed!")
    print("Graphs saved inside:", OUTPUT_DIR)
