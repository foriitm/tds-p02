#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "seaborn>=0.12.0",
#     "matplotlib>=3.7.0",
#     "scikit-learn>=1.3.0",
#     "httpx>=0.24.0",
#     "tenacity>=8.2.0",
#     "python-dotenv>=1.0.0",
#     "statsmodels>=0.14.0",
#     "pillow>=10.0.0",  # For image optimization
# ]
# ///

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition, cluster
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx
from dotenv import load_dotenv
import statsmodels.api as sm
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable is required")

AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai"
#AIPROXY_URL = "https://api.openai.com"
MAX_RETRIES = 3
CHART_SIZE = (10, 6)
DPI = 100
RANDOM_STATE = 42

# Configure plotting settings
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['figure.figsize'] = CHART_SIZE
plt.rcParams['savefig.dpi'] = DPI

# Suppress warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, (np.bool_, bool)):  # Handle boolean values
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class DataAnalyzer:
    def __init__(self, csv_path: str):
        """Initialize the analyzer with a CSV file path."""
        self.csv_path = Path(csv_path)
        self.output_dir = Path.cwd()
        self.df = None
        self.analysis_results = {}
        self.charts = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.errors = []  # Track errors for reporting
        
        # Set up logging for this instance
        self.logger = logging.getLogger(f"{__name__}.{self.csv_path.stem}")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler for this dataset
        fh = logging.FileHandler(self.output_dir / "analysis.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        
    def _validate_dataframe(self) -> None:
        """Validate the loaded dataframe."""
        if self.df.empty:
            raise ValueError("DataFrame is empty")
            
        if self.df.shape[1] == 1:
            self.logger.warning("DataFrame has only one column. Limited analysis possible.")
            
        if not any(self.numeric_cols):
            self.logger.warning("No numeric columns found. Limited analysis possible.")
            
        # Check for high cardinality in categorical columns
        for col in self.categorical_cols:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.5:
                self.logger.warning(f"Column {col} has high cardinality ({unique_ratio:.1%} unique values)")
                
        # Check for highly correlated features
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr().abs()
            high_corr = np.where(np.triu(corr_matrix, 1) > 0.95)
            for i, j in zip(*high_corr):
                self.logger.warning(
                    f"High correlation ({corr_matrix.iloc[i, j]:.2f}) between "
                    f"{self.numeric_cols[i]} and {self.numeric_cols[j]}"
                )
                
    def _clean_data(self) -> None:
        """Clean the dataframe."""
        # Remove columns with too many missing values
        missing_ratios = self.df.isnull().mean()
        cols_to_drop = missing_ratios[missing_ratios > 0.5].index
        if not cols_to_drop.empty:
            self.logger.warning(f"Dropping columns with >50% missing values: {list(cols_to_drop)}")
            self.df.drop(columns=cols_to_drop, inplace=True)
            
        # Remove duplicate rows
        dups = self.df.duplicated()
        if dups.any():
            self.logger.warning(f"Removing {dups.sum()} duplicate rows")
            self.df.drop_duplicates(inplace=True)
            
        # Handle infinite values
        inf_cols = self.df.isin([np.inf, -np.inf]).any()
        inf_cols = inf_cols[inf_cols].index
        if not inf_cols.empty:
            self.logger.warning(f"Replacing infinite values in columns: {list(inf_cols)}")
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
    def load_data(self) -> None:
        """Load and validate the CSV data."""
        try:
            # Try different encodings in order of likelihood
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    # Try to infer date columns
                    date_parser = lambda x: pd.to_datetime(x, errors='ignore')
                    self.df = pd.read_csv(self.csv_path, parse_dates=True, date_parser=date_parser, encoding=encoding)
                    self.logger.info(f"Successfully loaded dataset with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error loading CSV file with {encoding} encoding: {e}")
                    continue
            
            if self.df is None:
                raise ValueError("Could not load file with any supported encoding")
                
            self.logger.info(f"Loaded dataset with shape: {self.df.shape}")
            
            # Identify numeric and categorical columns
            self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = self.df.select_dtypes(include=['object', 'category', 'datetime64']).columns.tolist()
            
            # Clean and validate data
            self._clean_data()
            self._validate_dataframe()
            
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            raise
            
    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_llm(self, messages: List[Dict[str, str]], functions: Optional[List[Dict]] = None) -> Dict:
        """Call the LLM with retry logic."""
        try:
            headers = {
                "Authorization": f"Bearer {AIPROXY_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.7,
            }
            if functions:
                payload["functions"] = functions
                
            response = httpx.post(
                f"{AIPROXY_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            # Check for rate limiting
            if response.status_code == 429:
                self.logger.warning("Rate limited by API. Retrying...")
                raise Exception("Rate limited")
                
            return response.json()
            
        except httpx.TimeoutException:
            self.logger.warning("API request timed out. Retrying...")
            raise
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error occurred: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        stats = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_summary": self.df[self.numeric_cols].describe().to_dict() if self.numeric_cols else {},
            "categorical_summary": {
                col: self.df[col].value_counts().head().to_dict()
                for col in self.categorical_cols
            }
        }
        return stats

    def detect_outliers(self) -> Dict[str, Any]:
        """Detect outliers in numeric columns using IQR method."""
        outliers = {}
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = {
                "count": len(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]),
                "percentage": len(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]) / len(self.df) * 100,
                "bounds": {"lower": lower_bound, "upper": upper_bound}
            }
        return outliers

    def analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        if len(self.numeric_cols) < 2:
            return {}
            
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find strongest correlations
        strong_correlations = []
        for i in range(len(self.numeric_cols)):
            for j in range(i + 1, len(self.numeric_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:  # Threshold for strong correlation
                    strong_correlations.append({
                        "var1": self.numeric_cols[i],
                        "var2": self.numeric_cols[j],
                        "correlation": corr
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations
        }

    def perform_clustering(self) -> Dict[str, Any]:
        """Perform basic clustering analysis on numeric data."""
        if len(self.numeric_cols) < 2:
            return {}
            
        # Prepare data
        X = self.df[self.numeric_cols].copy()
        X = X.fillna(X.mean())  # Handle missing values
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        max_clusters = min(10, len(self.df) // 2)
        for k in range(2, max_clusters + 1):
            kmeans = cluster.KMeans(n_clusters=k, random_state=RANDOM_STATE)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        optimal_clusters = 2  # default
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            diffs_of_diffs = np.diff(diffs)
            elbow_idx = np.argmax(diffs_of_diffs) + 2
            optimal_clusters = elbow_idx + 2
        
        # Perform clustering with optimal number of clusters
        kmeans = cluster.KMeans(n_clusters=optimal_clusters, random_state=RANDOM_STATE)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Get cluster centers
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        return {
            "optimal_clusters": optimal_clusters,
            "cluster_sizes": pd.Series(clusters).value_counts().to_dict(),
            "cluster_centers": {
                f"cluster_{i}": {col: center for col, center in zip(self.numeric_cols, centers[i])}
                for i in range(optimal_clusters)
            },
            "inertia_values": inertias
        }

    def detect_patterns(self) -> Dict[str, Any]:
        """Detect patterns in the data using various techniques."""
        patterns = {}
        
        # Time series detection
        date_cols = [col for col in self.df.columns if 
                    self.df[col].dtype in ['datetime64[ns]', 'object'] and 
                    pd.to_datetime(self.df[col], errors='coerce').notna().any()]
        
        if date_cols:
            patterns["time_series"] = self._analyze_time_series(date_cols[0])
            
        # Geographic data detection
        geo_cols = [col for col in self.categorical_cols if 
                   any(geo_term in col.lower() for geo_term in 
                       ['country', 'city', 'state', 'region', 'location'])]
        
        if geo_cols:
            patterns["geographic"] = {
                col: self.df[col].value_counts().head(10).to_dict()
                for col in geo_cols
            }
            
        return patterns

    def _analyze_time_series(self, date_col: str) -> Dict[str, Any]:
        """Analyze time series patterns in the data."""
        try:
            df_temp = self.df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.sort_values(date_col)
            
            # Basic time series metrics
            metrics = {
                "start_date": df_temp[date_col].min().strftime("%Y-%m-%d"),
                "end_date": df_temp[date_col].max().strftime("%Y-%m-%d"),
                "time_span_days": (df_temp[date_col].max() - df_temp[date_col].min()).days
            }
            
            # Check for seasonality if we have enough data points
            if len(df_temp) >= 30 and any(self.numeric_cols):
                target_col = self.numeric_cols[0]  # Use first numeric column as example
                series = df_temp[target_col].fillna(method='ffill')
                decomposition = sm.tsa.seasonal_decompose(
                    series,
                    period=min(30, len(df_temp) // 2)
                )
                metrics["seasonality"] = {
                    "trend": decomposition.trend.dropna().tolist()[-5:],  # Last 5 points
                    "seasonal": decomposition.seasonal.dropna().tolist()[:5],  # First 5 points
                    "resid": float(decomposition.resid.dropna().std())
                }
            
            return metrics
        except Exception as e:
            self.logger.warning(f"Time series analysis failed: {e}")
            return {}

    def analyze(self) -> None:
        """Main analysis pipeline."""
        self.load_data()
        
        # Perform all analyses
        self.analysis_results = {
            "basic_stats": self.get_basic_stats(),
            "outliers": self.detect_outliers(),
            "correlations": self.analyze_correlations(),
            "clustering": self.perform_clustering(),
            "patterns": self.detect_patterns()
        }
        
        # Get LLM insights
        self._get_llm_insights()
        
    def _get_llm_insights(self) -> None:
        """Get insights from LLM based on analysis results."""
        # Create a concise summary of the analysis
        summary = {
            "dataset_info": {
                "rows": int(self.df.shape[0]),  # Convert numpy.int64 to int
                "columns": int(self.df.shape[1]),  # Convert numpy.int64 to int
                "column_types": {
                    "numeric": len(self.numeric_cols),
                    "categorical": len(self.categorical_cols)
                }
            },
            "key_findings": {
                "missing_values": any(self.analysis_results["basic_stats"]["missing_values"].values()),
                "outliers_detected": any(info["count"] > 0 for info in self.analysis_results["outliers"].values()),
                "strong_correlations": len(self.analysis_results["correlations"].get("strong_correlations", [])),
                "clusters_found": self.analysis_results["clustering"].get("optimal_clusters", 0) if self.analysis_results["clustering"] else 0,
                "patterns": list(self.analysis_results["patterns"].keys()) if self.analysis_results["patterns"] else []
            }
        }
        
        messages = [
            {
                "role": "system",
                "content": "You are a data analysis expert. Analyze the following results and provide key insights."
            },
            {
                "role": "user",
                "content": f"Here is the analysis of the dataset {self.csv_path.name}:\n{json.dumps(summary, cls=NumpyEncoder, indent=2)}"
            }
        ]
        
        try:
            response = self.call_llm(messages)
            self.analysis_results["llm_insights"] = response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to get LLM insights: {e}")
            self.analysis_results["llm_insights"] = "Failed to generate insights."

    def visualize(self) -> None:
        """Create visualizations."""
        # Create and save visualizations
        self._plot_correlation_heatmap()
        self._plot_outliers()
        self._plot_clustering()
        
        # Close all figures to free memory
        plt.close('all')
        
    def _truncate_label(self, label: str, max_length: int = 20) -> str:
        """Truncate long labels and add ellipsis."""
        if len(label) > max_length:
            return label[:max_length-3] + "..."
        return label

    def _optimize_png_size(self, png_path: Path) -> None:
        """Ensure PNG files are optimally sized (around 512x512 pixels)."""
        try:
            # Open image
            with Image.open(png_path) as img:
                # Calculate scaling factor to get closest to 512x512
                max_size = 512
                scale = min(max_size / img.width, max_size / img.height)
                
                if scale < 1:  # Only resize if image is too large
                    new_size = (int(img.width * scale), int(img.height * scale))
                    resized = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Save with optimal compression
                    resized.save(png_path, "PNG", optimize=True)
                    self.logger.info(f"Optimized {png_path.name} to {new_size}")

        except Exception as e:
            self.logger.warning(f"Failed to optimize {png_path}: {e}")

    def _save_plot(self, filename: str, **kwargs) -> None:
        """Save plot with consistent settings and optimization."""
        filepath = self.output_dir / filename
        plt.savefig(
            filepath,
            bbox_inches='tight',
            dpi=DPI,
            pad_inches=kwargs.get('pad_inches', 0.5)
        )
        self._optimize_png_size(filepath)
        self.charts.append(filename)

    def _plot_correlation_heatmap(self) -> None:
        """Create a correlation heatmap for numeric columns."""
        if not self.numeric_cols or len(self.numeric_cols) < 2:
            return
            
        # Get correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # If too many features, select only the most important ones
        n_features = len(self.numeric_cols)
        if n_features > 15:  # If more than 15 features, select most correlated ones
            # Get average absolute correlation for each feature
            mean_abs_corr = abs(corr_matrix).mean()
            # Select top 15 features with highest average correlation
            top_features = mean_abs_corr.nlargest(15).index
            corr_matrix = corr_matrix.loc[top_features, top_features]
            self.logger.info(f"Selected top {len(top_features)} most correlated features for visualization")
        
        # Calculate figure size based on number of features
        # Minimum size 8x6, then scale up based on number of features
        size_scale = max(1.0, n_features / 10)  # Scale factor based on features
        fig_size = (
            max(8, min(CHART_SIZE[0] * size_scale, 20)),  # Max width 20
            max(6, min(CHART_SIZE[1] * size_scale, 16))   # Max height 16
        )
        
        # Create figure
        fig = plt.figure(figsize=fig_size)
        
        # Truncate long column names
        max_label_length = max(20, 50 // n_features)  # Adjust max length based on number of features
        truncated_cols = [self._truncate_label(col, max_label_length) for col in corr_matrix.columns]
        
        # Create heatmap with adjusted font size
        font_size = max(6, min(10, 120 / n_features))  # Scale font size based on number of features
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": min(1.0, 5/n_features)},  # Adjust colorbar size
            xticklabels=truncated_cols,
            yticklabels=truncated_cols,
            annot_kws={'size': font_size}  # Adjust annotation font size
        )
        
        plt.title('Correlation Heatmap of Numeric Features', pad=20, fontsize=font_size * 1.5)
        
        # Rotate labels and adjust font size
        plt.xticks(rotation=45, ha='right', fontsize=font_size)
        plt.yticks(rotation=0, fontsize=font_size)
        
        # Adjust layout with different padding based on figure size
        padding = max(0.5, 2.0 / size_scale)  # Reduce padding for larger matrices
        plt.tight_layout(pad=padding)
        
        # Save plot with optimization
        self._save_plot('correlation_heatmap.png', pad_inches=max(0.2, 0.5 / size_scale))
        plt.close(fig)
        
    def _plot_outliers(self) -> None:
        """Create box plots to visualize outliers in numeric columns."""
        if not self.numeric_cols:
            return
            
        # Select top 5 columns with most outliers
        outlier_counts = {
            col: self.analysis_results["outliers"][col]["count"]
            for col in self.numeric_cols
        }
        top_cols = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if not top_cols:
            return
            
        # Create figure with seaborn
        # Adjust figure size based on label lengths
        max_label_length = max(len(col) for col, _ in top_cols)
        fig_width = max(CHART_SIZE[0], max_label_length * 0.25)
        fig = plt.figure(figsize=(fig_width, CHART_SIZE[1]))
        
        # Create box plots with truncated labels
        plot_data = self.df[[col for col, _ in top_cols]]
        plot_data.columns = [self._truncate_label(col) for col, _ in top_cols]
        sns.boxplot(data=plot_data)
        
        plt.title('Distribution and Outliers of Top Numeric Features', pad=20)
        
        # Rotate and align the tick labels so they look better
        plt.xticks(rotation=45, ha='right')
        
        # Use tight_layout with custom padding
        plt.tight_layout(pad=2.0)
        
        # Save plot with optimization
        self._save_plot('outliers_boxplot.png', pad_inches=0.5)
        plt.close(fig)
        
    def _plot_clustering(self) -> None:
        """Visualize clustering results using PCA for dimensionality reduction."""
        if not self.numeric_cols or len(self.numeric_cols) < 2:
            return
            
        # Prepare data
        X = self.df[self.numeric_cols].copy()
        X = X.fillna(X.mean())
        
        # Standardize features
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = decomposition.PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Perform clustering
        n_clusters = self.analysis_results["clustering"]["optimal_clusters"]
        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create figure with seaborn
        fig = plt.figure(figsize=CHART_SIZE)
        
        # Create scatter plot
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=clusters,
            cmap='viridis',
            alpha=0.6
        )
        
        # Add cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            c='red',
            marker='x',
            s=200,
            linewidths=3,
            label='Cluster Centers'
        )
        
        # Format variance percentages
        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        
        plt.title(f'Clustering Results (k={n_clusters})', pad=20)
        plt.xlabel(f'First Principal Component ({var1:.1f}% variance)')
        plt.ylabel(f'Second Principal Component ({var2:.1f}% variance)')
        
        # Add colorbar with a better size
        cbar = plt.colorbar(scatter, label='Cluster')
        cbar.ax.set_ylabel('Cluster', rotation=270, labelpad=15)
        
        # Add legend with a good position
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Use tight_layout with custom padding
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for colorbar
        
        # Save plot with optimization
        self._save_plot('clustering_pca.png', pad_inches=0.5)
        plt.close(fig)

    def _analyze_chart_with_vision(self, chart_path: str) -> str:
        """Analyze a chart using vision capabilities."""
        try:
            # Read image file as base64
            with open(chart_path, "rb") as image_file:
                import base64
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a data visualization expert. Analyze the chart and provide insights about what it reveals."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this chart and describe what insights it reveals about the data:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ]
            
            response = self.call_llm(messages)
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Failed to analyze chart with vision: {e}")
            return "Failed to analyze chart."

    def _get_analysis_functions(self) -> List[Dict]:
        """Define functions that can be called by the LLM."""
        return [
            {
                "name": "analyze_column_distribution",
                "description": "Analyze the distribution of a specific column",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column_name": {
                            "type": "string",
                            "description": "The name of the column to analyze"
                        }
                    },
                    "required": ["column_name"]
                }
            },
            {
                "name": "find_correlations",
                "description": "Find correlations between a target column and other numeric columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_column": {
                            "type": "string",
                            "description": "The target column to find correlations with"
                        }
                    },
                    "required": ["target_column"]
                }
            },
            {
                "name": "suggest_visualization",
                "description": "Suggest appropriate visualization types for specific columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of column names to visualize"
                        }
                    },
                    "required": ["columns"]
                }
            }
        ]

    def _handle_function_call(self, function_call: Dict) -> Dict:
        """Handle function calls from the LLM."""
        function_name = function_call["name"]
        arguments = json.loads(function_call["arguments"])
        
        if function_name == "analyze_column_distribution":
            column_name = arguments["column_name"]
            if column_name not in self.df.columns:
                return {"error": f"Column {column_name} not found"}
                
            if column_name in self.numeric_cols:
                stats = self.df[column_name].describe().to_dict()
                return {
                    "type": "numeric",
                    "statistics": stats,
                    "has_outliers": self.analysis_results["outliers"].get(column_name, {}).get("count", 0) > 0
                }
            else:
                value_counts = self.df[column_name].value_counts().head(10).to_dict()
                return {
                    "type": "categorical",
                    "value_counts": value_counts,
                    "unique_values": self.df[column_name].nunique()
                }
                
        elif function_name == "find_correlations":
            target_column = arguments["target_column"]
            if target_column not in self.numeric_cols:
                return {"error": f"Column {target_column} is not numeric"}
                
            correlations = self.df[self.numeric_cols].corr()[target_column].sort_values(ascending=False).to_dict()
            return {
                "correlations": {k: v for k, v in correlations.items() if k != target_column and abs(v) > 0.3}
            }
            
        elif function_name == "suggest_visualization":
            columns = arguments["columns"]
            invalid_cols = [col for col in columns if col not in self.df.columns]
            if invalid_cols:
                return {"error": f"Columns not found: {invalid_cols}"}
                
            suggestions = []
            if len(columns) == 1:
                col = columns[0]
                if col in self.numeric_cols:
                    suggestions.append({"type": "histogram", "description": "Show distribution of values"})
                    suggestions.append({"type": "box plot", "description": "Show outliers and quartiles"})
                else:
                    suggestions.append({"type": "bar chart", "description": "Show frequency of categories"})
                    suggestions.append({"type": "pie chart", "description": "Show proportion of categories"})
            elif len(columns) == 2:
                if all(col in self.numeric_cols for col in columns):
                    suggestions.append({"type": "scatter plot", "description": "Show relationship between variables"})
                    suggestions.append({"type": "hexbin plot", "description": "Show density of points"})
                elif any(col in self.numeric_cols for col in columns):
                    suggestions.append({"type": "box plot", "description": "Show distribution by category"})
                    suggestions.append({"type": "violin plot", "description": "Show detailed distribution by category"})
                else:
                    suggestions.append({"type": "heatmap", "description": "Show relationship between categories"})
                    suggestions.append({"type": "stacked bar chart", "description": "Show composition of categories"})
            
            return {"suggestions": suggestions}
            
        return {"error": f"Unknown function {function_name}"}

    def generate_story(self) -> None:
        """Generate the narrative and create README.md."""
        # First analyze charts with vision
        chart_insights = {}
        for chart in self.charts:
            chart_path = self.output_dir / chart
            if chart_path.exists():
                chart_insights[chart] = self._analyze_chart_with_vision(chart_path)
        
        # Prepare the context for the LLM
        context = {
            "filename": self.csv_path.name,
            "dataset_info": {
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "numeric_features": len(self.numeric_cols),
                "categorical_features": len(self.categorical_cols),
                "dataset_types": self.analysis_results.get("dataset_types", [])
            },
            "key_findings": {
                "basic_stats": self._get_key_stats(),
                "patterns": self._get_key_patterns(),
                "relationships": self._get_key_relationships()
            },
            "charts": {name: insights for name, insights in chart_insights.items()},
            "implications": self._get_implications()
        }
        
        # Create system message with detailed instructions
        system_message = """You are a data analysis expert and storyteller. Create an engaging, insightful narrative about this dataset analysis.

Follow this specific structure:
1. Introduction
   - Brief dataset overview
   - Key characteristics
   - Analysis scope

2. Methodology
   - Analysis techniques used
   - Why these techniques were chosen
   - How they complement each other

3. Key Findings
   - Most significant discoveries
   - Statistical evidence
   - Visual insights from charts
   - Unexpected patterns

4. Implications
   - What these findings mean
   - Actionable insights
   - Potential applications
   - Areas for further investigation

Style Guidelines:
- Use clear, professional language
- Include specific numbers and statistics
- Reference visualizations naturally in the text
- Use Markdown formatting effectively
- Highlight critical insights with bold text
- Use bullet points for lists
- Keep paragraphs focused and concise

Remember to:
- Connect findings to real-world implications
- Explain technical concepts clearly
- Emphasize practical applications
- Maintain a logical flow of ideas"""

        # Create user message with structured context
        user_message = f"""Analyze this dataset: {context['filename']}

Dataset Characteristics:
- {context['dataset_info']['rows']} rows, {context['dataset_info']['columns']} columns
- {context['dataset_info']['numeric_features']} numeric features
- {context['dataset_info']['categorical_features']} categorical features
- Dataset types: {', '.join(context['dataset_info']['dataset_types'])}

Key Findings:
{json.dumps(context['key_findings'], cls=NumpyEncoder, indent=2)}

Visual Analysis:
{json.dumps(context['charts'], cls=NumpyEncoder, indent=2)}

Implications:
{json.dumps(context['implications'], cls=NumpyEncoder, indent=2)}

Create a comprehensive README.md that tells this data's story, following the structure in the system message."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        try:
            # Get the narrative from LLM with function calling
            response = self.call_llm(messages, functions=self._get_analysis_functions())
            
            # Handle any function calls
            while response["choices"][0]["message"].get("function_call"):
                function_call = response["choices"][0]["message"]["function_call"]
                function_response = self._handle_function_call(function_call)
                
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call
                })
                messages.append({
                    "role": "function",
                    "name": function_call["name"],
                    "content": json.dumps(function_response, cls=NumpyEncoder)
                })
                
                response = self.call_llm(messages, functions=self._get_analysis_functions())
            
            narrative = response["choices"][0]["message"]["content"]
            
            # Save the README.md
            readme_path = self.output_dir / 'README.md'
            with open(readme_path, 'w') as f:
                f.write(narrative)
                
            self.logger.info(f"Generated README.md at {readme_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate story: {e}")
            raise

    def _get_key_stats(self) -> Dict[str, Any]:
        """Extract key statistics for the narrative."""
        stats = {}
        
        if "basic_stats" in self.analysis_results:
            numeric_summary = self.analysis_results["basic_stats"].get("numeric_summary", {})
            stats["numeric_insights"] = {
                col: {
                    "mean": float(summary.get("mean", 0)),
                    "std": float(summary.get("std", 0)),
                    "range": [float(summary.get("min", 0)), float(summary.get("max", 0))]
                }
                for col, summary in numeric_summary.items()
            }
        
        if "outliers" in self.analysis_results:
            stats["outlier_summary"] = {
                col: {
                    "count": info["count"],
                    "percentage": round(info["percentage"], 2)
                }
                for col, info in self.analysis_results["outliers"].items()
                if info["count"] > 0
            }
            
        return stats

    def _get_key_patterns(self) -> Dict[str, Any]:
        """Extract key patterns for the narrative."""
        patterns = {}
        
        # Get clustering insights if available
        if "clustering" in self.analysis_results:
            clustering = self.analysis_results["clustering"]
            if clustering:
                patterns["clusters"] = {
                    "count": clustering.get("optimal_clusters", 0),
                    "sizes": clustering.get("cluster_sizes", {})
                }
        
        # Get dimensionality insights if available
        if "dimensionality" in self.analysis_results:
            dim = self.analysis_results["dimensionality"]
            if dim:
                patterns["dimensionality"] = {
                    "components_95": dim.get("components_needed", {}).get("for_95_percent_variance", 0),
                    "reduction_potential": dim.get("dimensionality_reduction_potential", {})
                }
        
        return patterns

    def _get_key_relationships(self) -> Dict[str, Any]:
        """Extract key relationships for the narrative."""
        relationships = {}
        
        # Get correlation insights if available
        if "correlations" in self.analysis_results:
            correlations = self.analysis_results["correlations"]
            if correlations and "strong_correlations" in correlations:
                relationships["strong_correlations"] = [
                    {
                        "variables": [corr["var1"], corr["var2"]],
                        "strength": round(corr["correlation"], 3)
                    }
                    for corr in correlations["strong_correlations"]
                ]
        
        return relationships

    def _get_implications(self) -> Dict[str, Any]:
        """Generate implications based on the analysis results."""
        implications = {}
        
        # Add predictive potential implications
        if "predictive_power" in self.analysis_results:
            pred = self.analysis_results["predictive_power"]
            if pred:
                implications["predictive_potential"] = {
                    "suitable_for_ml": pred.get("overall_assessment", {}).get("recommendation", {}).get("suitable_for_ml", False),
                    "suggested_approaches": pred.get("overall_assessment", {}).get("recommendation", {}).get("suggested_approaches", [])
                }
        
        # Add data quality implications
        implications["data_quality"] = {
            "completeness": 1 - self.df.isnull().mean().mean(),
            "consistency": len(self.errors) == 0,
            "recommendations": self._get_quality_recommendations()
        }
        
        return implications

    def _get_quality_recommendations(self) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        # Check for missing values
        if self.df.isnull().any().any():
            recommendations.append("Consider handling missing values")
            
        # Check for high cardinality
        for col in self.categorical_cols:
            if self.df[col].nunique() / len(self.df) > 0.5:
                recommendations.append(f"High cardinality in {col} might need grouping")
                
        # Check for imbalance in categorical columns
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > 0.8:  # If dominant category > 80%
                recommendations.append(f"Consider addressing imbalance in {col}")
                
        return recommendations

    def safe_generate_story(self) -> None:
        """Safely generate the narrative, handling all possible errors."""
        try:
            # First try to generate the full story with LLM
            self.generate_story()
        except Exception as e:
            self.logger.error(f"Failed to generate full story: {e}")
            self.errors.append(f"Story generation failed: {str(e)}")
            
            # Create a basic but informative README
            basic_readme = f"""# Analysis of {self.csv_path.name}

## Dataset Information
- Rows: {self.df.shape[0] if self.df is not None else 'N/A'}
- Columns: {self.df.shape[1] if self.df is not None else 'N/A'}
- Numeric Features: {len(self.numeric_cols)}
- Categorical Features: {len(self.categorical_cols)}

## Analysis Results
"""
            # Add detected dataset types
            if "dataset_types" in self.analysis_results:
                basic_readme += "\n### Dataset Types\n"
                for dtype in self.analysis_results["dataset_types"]:
                    basic_readme += f"- {dtype}\n"

            # Add whatever analysis results we have
            for analysis_type, results in self.analysis_results.items():
                if results and analysis_type != "dataset_types":  # Skip dataset_types as we've already added it
                    basic_readme += f"\n### {analysis_type.replace('_', ' ').title()}\n"
                    if isinstance(results, dict):
                        for key, value in results.items():
                            if not isinstance(value, (dict, list)):  # Only show simple values
                                basic_readme += f"- {key}: {value}\n"
                    else:
                        basic_readme += f"{results}\n"

            # Add any charts that were successfully generated
            if self.charts:
                basic_readme += "\n## Visualizations\n"
                for chart in self.charts:
                    basic_readme += f"\n![{chart}]({chart})\n"

            # Add error summary if there were any errors
            if self.errors:
                basic_readme += "\n## Analysis Issues\n"
                basic_readme += "The following issues were encountered during analysis:\n"
                for error in self.errors:
                    basic_readme += f"- {error}\n"

            # Save the basic README
            readme_path = self.output_dir / 'README.md'
            with open(readme_path, 'w') as f:
                f.write(basic_readme)
            
            self.logger.info(f"Generated basic README.md at {readme_path}")

    def _detect_dataset_type(self) -> str:
        """Detect the type of dataset to determine appropriate analyses."""
        # Check for time series
        date_cols = [col for col in self.df.columns if 
                    self.df[col].dtype in ['datetime64[ns]', 'object'] and 
                    pd.to_datetime(self.df[col], errors='coerce').notna().mean() > 0.5]
        
        # Check for geographic data
        geo_cols = [col for col in self.categorical_cols if 
                   any(geo_term in col.lower() for geo_term in 
                       ['country', 'city', 'state', 'region', 'location', 'lat', 'long', 'latitude', 'longitude'])]
        
        # Check for text data
        text_cols = [col for col in self.categorical_cols if 
                    self.df[col].dtype == 'object' and 
                    self.df[col].str.len().mean() > 50]  # Average length > 50 chars
        
        # Check for categorical target (classification)
        potential_targets = [col for col in self.categorical_cols if 
                           self.df[col].nunique() < 10 and  # Few unique values
                           'id' not in col.lower()]  # Not an ID column
        
        # Check for numeric target (regression)
        numeric_targets = [col for col in self.numeric_cols if 
                         'id' not in col.lower() and
                         'index' not in col.lower() and
                         self.df[col].nunique() > 10]  # Many unique values
        
        dataset_types = []
        if date_cols:
            dataset_types.append("time_series")
        if geo_cols:
            dataset_types.append("geographic")
        if text_cols:
            dataset_types.append("text_heavy")
        if potential_targets:
            dataset_types.append("classification")
        if numeric_targets:
            dataset_types.append("regression")
        if len(self.numeric_cols) > 5:
            dataset_types.append("high_dimensional")
            
        self.logger.info(f"Detected dataset types: {dataset_types}")
        return dataset_types

    def _get_relevant_analyses(self, dataset_types: List[str]) -> List[tuple]:
        """Get relevant analyses based on dataset type."""
        # Base analyses for all datasets
        analyses = [
            ("basic_stats", self.get_basic_stats),
            ("outliers", self.detect_outliers),
            ("correlations", self.analyze_correlations)
        ]
        
        if "high_dimensional" in dataset_types:
            analyses.extend([
                ("clustering", self.perform_clustering),
                ("dimensionality", self._analyze_dimensionality)
            ])
            
        if "time_series" in dataset_types:
            analyses.extend([
                ("seasonality", self._analyze_seasonality),
                ("trend", self._analyze_trend),
                ("forecasting", self._analyze_forecast_potential)
            ])
            
        if "geographic" in dataset_types:
            analyses.extend([
                ("spatial", self._analyze_spatial_patterns),
                ("regional", self._analyze_regional_stats)
            ])
            
        if "text_heavy" in dataset_types:
            analyses.extend([
                ("text", self._analyze_text_features),
                ("sentiment", self._analyze_sentiment)
            ])
            
        if "classification" in dataset_types or "regression" in dataset_types:
            analyses.extend([
                ("feature_importance", self._analyze_feature_importance),
                ("predictive_power", self._analyze_predictive_potential)
            ])
            
        return analyses

    def safe_analyze(self) -> None:
        """Safely run all analyses, catching and logging errors."""
        try:
            self.load_data()
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            self.errors.append(f"Data loading failed: {str(e)}")
            return

        # Detect dataset types and get relevant analyses
        dataset_types = self._detect_dataset_type()
        analyses = self._get_relevant_analyses(dataset_types)
        
        # Store dataset types in results
        self.analysis_results["dataset_types"] = dataset_types

        for name, func in analyses:
            try:
                self.analysis_results[name] = func()
                self.logger.info(f"Completed {name} analysis")
            except Exception as e:
                self.logger.error(f"Failed to complete {name} analysis: {e}")
                self.errors.append(f"{name} analysis failed: {str(e)}")
                self.analysis_results[name] = {}

    def safe_visualize(self) -> None:
        """Safely create visualizations, catching and logging errors."""
        if self.df is None:
            self.logger.error("Cannot create visualizations: No data loaded")
            self.errors.append("Visualization failed: No data loaded")
            return

        # Get dataset types
        dataset_types = self.analysis_results.get("dataset_types", [])
        
        # Select most relevant visualizations based on dataset type
        visualizations = []
        
        # Always include correlation heatmap if we have numeric columns
        if len(self.numeric_cols) >= 2:
            visualizations.append(("correlation_heatmap", self._plot_correlation_heatmap))
        
        # Add specialized visualizations based on dataset type (max 2 more)
        if "high_dimensional" in dataset_types:
            visualizations.append(("clustering_pca", self._plot_clustering))
        elif "time_series" in dataset_types:
            visualizations.append(("seasonality", self._plot_seasonality))
        elif "geographic" in dataset_types:
            visualizations.append(("spatial_distribution", self._plot_spatial_distribution))
        elif "text_heavy" in dataset_types:
            visualizations.append(("text_length_dist", self._plot_text_distribution))
        elif "classification" in dataset_types:
            visualizations.append(("class_distribution", self._plot_class_distribution))
        elif "regression" in dataset_types:
            visualizations.append(("target_distribution", self._plot_target_distribution))
            
        # If we still have room for one more chart and have numeric columns, add outliers
        if len(visualizations) < 3 and self.numeric_cols:
            visualizations.append(("outliers_boxplot", self._plot_outliers))

        # Ensure we have at least one visualization
        if not visualizations and self.numeric_cols:
            visualizations.append(("outliers_boxplot", self._plot_outliers))

        # Clear any existing charts
        self.charts = []

        # Generate selected visualizations
        for name, func in visualizations:
            try:
                func()
                self.logger.info(f"Created {name} visualization")
            except Exception as e:
                self.logger.error(f"Failed to create {name} visualization: {e}")
                self.errors.append(f"{name} visualization failed: {str(e)}")

        plt.close('all')  # Clean up any remaining figures

    def _plot_variance_explained(self) -> None:
        """Plot cumulative explained variance from PCA."""
        if len(self.numeric_cols) < 3:
            return
            
        # Prepare data
        X = self.df[self.numeric_cols].copy()
        X = X.fillna(X.mean())
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = decomposition.PCA()
        pca.fit(X_scaled)
        
        # Create plot
        fig = plt.figure(figsize=CHART_SIZE)
        
        # Plot cumulative explained variance
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                np.cumsum(pca.explained_variance_ratio_),
                'bo-')
        
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
        plt.grid(True)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        
        self._save_plot('variance_explained.png')
        plt.close(fig)

    def _plot_seasonality(self) -> None:
        """Plot seasonal patterns in time series data."""
        date_cols = [col for col in self.df.columns if 
                    self.df[col].dtype in ['datetime64[ns]', 'object'] and 
                    pd.to_datetime(self.df[col], errors='coerce').notna().mean() > 0.5]
        
        if not date_cols or not self.numeric_cols:
            return
            
        date_col = date_cols[0]
        df_temp = self.df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        
        # Select top numeric column by variance
        num_col = self.df[self.numeric_cols].var().nlargest(1).index[0]
        
        # Create plot
        fig = plt.figure(figsize=CHART_SIZE)
        
        # Plot time series decomposition
        series = df_temp.set_index(date_col)[num_col].resample('D').mean().interpolate()
        decomposition = sm.tsa.seasonal_decompose(
            series,
            period=min(30, len(series) // 2)
        )
        
        plt.subplot(411)
        plt.plot(series.index, series.values)
        plt.title(f'Time Series Decomposition: {num_col}')
        plt.ylabel('Observed')
        
        plt.subplot(412)
        plt.plot(series.index, decomposition.trend)
        plt.ylabel('Trend')
        
        plt.subplot(413)
        plt.plot(series.index, decomposition.seasonal)
        plt.ylabel('Seasonal')
        
        plt.subplot(414)
        plt.plot(series.index, decomposition.resid)
        plt.ylabel('Residual')
        
        plt.tight_layout()
        self._save_plot('seasonality.png')
        plt.close(fig)

    def _plot_spatial_distribution(self) -> None:
        """Plot spatial distribution of data points."""
        # Find geographic columns
        lat_cols = [col for col in self.numeric_cols if 
                   any(term in col.lower() for term in ['lat', 'latitude'])]
        long_cols = [col for col in self.numeric_cols if 
                    any(term in col.lower() for term in ['long', 'longitude'])]
        
        if not lat_cols or not long_cols:
            return
            
        lat_col, long_col = lat_cols[0], long_cols[0]
        
        # Create plot
        fig = plt.figure(figsize=CHART_SIZE)
        
        # Plot points
        plt.scatter(
            self.df[long_col],
            self.df[lat_col],
            alpha=0.5,
            s=50
        )
        
        plt.title('Spatial Distribution of Data Points')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        
        # Add density contours if enough points
        if len(self.df) > 100:
            try:
                sns.kdeplot(
                    data=self.df,
                    x=long_col,
                    y=lat_col,
                    levels=5,
                    color='r',
                    alpha=0.3
                )
            except Exception:
                pass
        
        plt.tight_layout()
        self._save_plot('spatial_distribution.png')
        plt.close(fig)

    def _plot_text_distribution(self) -> None:
        """Plot distribution of text lengths and word counts."""
        text_cols = [col for col in self.categorical_cols if 
                    self.df[col].dtype == 'object' and 
                    self.df[col].str.len().mean() > 50]
        
        if not text_cols:
            return
            
        # Select first text column
        col = text_cols[0]
        text_series = self.df[col].dropna().astype(str)
        
        # Create plot
        fig = plt.figure(figsize=CHART_SIZE)
        
        # Plot length distribution
        plt.subplot(211)
        sns.histplot(text_series.str.len(), bins=50)
        plt.title(f'Text Length Distribution: {col}')
        plt.xlabel('Character Count')
        
        # Plot word count distribution
        plt.subplot(212)
        sns.histplot(text_series.str.split().str.len(), bins=50)
        plt.title('Word Count Distribution')
        plt.xlabel('Word Count')
        
        plt.tight_layout()
        self._save_plot('text_distribution.png')
        plt.close(fig)

    def _plot_class_distribution(self) -> None:
        """Plot distribution of classes for classification problems."""
        categorical_targets = [col for col in self.categorical_cols if 
                             self.df[col].nunique() < 10 and 
                             'id' not in col.lower()]
        
        if not categorical_targets:
            return
            
        target = categorical_targets[0]
        
        # Create plot
        fig = plt.figure(figsize=CHART_SIZE)
        
        # Plot class distribution
        class_dist = self.df[target].value_counts()
        sns.barplot(x=class_dist.index, y=class_dist.values)
        
        plt.title(f'Class Distribution: {target}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self._save_plot('class_distribution.png')
        plt.close(fig)

    def _plot_target_distribution(self) -> None:
        """Plot distribution of target variable for regression problems."""
        numeric_targets = [col for col in self.numeric_cols if 
                         'id' not in col.lower() and
                         'index' not in col.lower() and
                         self.df[col].nunique() > 10]
        
        if not numeric_targets:
            return
            
        target = numeric_targets[0]
        
        # Create plot
        fig = plt.figure(figsize=CHART_SIZE)
        
        # Plot target distribution
        sns.histplot(self.df[target].dropna(), bins=50)
        plt.title(f'Target Distribution: {target}')
        plt.xlabel('Value')
        
        # Add normal distribution fit
        from scipy import stats
        x = np.linspace(self.df[target].min(), self.df[target].max(), 100)
        params = stats.norm.fit(self.df[target].dropna())
        plt.plot(x, stats.norm.pdf(x, *params) * len(self.df) * (self.df[target].max() - self.df[target].min()) / 50,
                'r-', lw=2, label='Normal Fit')
        
        plt.legend()
        plt.tight_layout()
        self._save_plot('target_distribution.png')
        plt.close(fig)

    def _analyze_dimensionality(self) -> Dict[str, Any]:
        """Analyze dimensionality of the dataset using PCA."""
        if len(self.numeric_cols) < 3:
            return {}
            
        # Prepare data
        X = self.df[self.numeric_cols].copy()
        X = X.fillna(X.mean())
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA Analysis
        pca = decomposition.PCA()
        pca.fit(X_scaled)
        
        # Calculate metrics
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        
        # Get feature importance based on PCA components
        feature_importance = pd.DataFrame(
            abs(pca.components_[:3]),  # Top 3 components
            columns=self.numeric_cols
        ).mean()
        
        # Find highly correlated features
        corr_matrix = self.df[self.numeric_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(self.numeric_cols)):
            for j in range(i + 1, len(self.numeric_cols)):
                if corr_matrix.iloc[i, j] > 0.8:  # Threshold for high correlation
                    high_corr_pairs.append({
                        "feature1": self.numeric_cols[i],
                        "feature2": self.numeric_cols[j],
                        "correlation": float(corr_matrix.iloc[i, j])
                    })
        
        return {
            "total_features": len(self.numeric_cols),
            "components_needed": {
                "for_95_percent_variance": int(n_components_95),
                "for_90_percent_variance": int(n_components_90)
            },
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist()[:5],  # Top 5 components
            "cumulative_variance": cumulative_variance.tolist()[:5],  # Top 5 cumulative
            "top_features_by_importance": {
                name: float(importance)
                for name, importance in feature_importance.nlargest(5).items()
            },
            "dimensionality_reduction_potential": {
                "high": n_components_95 < len(self.numeric_cols) * 0.3,
                "reduction_ratio": float(n_components_95 / len(self.numeric_cols))
            },
            "high_correlation_groups": high_corr_pairs
        }

    def _analyze_text_features(self) -> Dict[str, Any]:
        """Analyze text features in the dataset."""
        text_cols = [col for col in self.categorical_cols if 
                    self.df[col].dtype == 'object' and 
                    self.df[col].str.len().mean() > 50]
        
        if not text_cols:
            return {}
            
        results = {}
        
        for col in text_cols[:3]:  # Analyze top 3 text columns
            text_series = self.df[col].dropna().astype(str)
            
            # Basic statistics
            word_counts = text_series.str.split().str.len()
            char_counts = text_series.str.len()
            
            # Calculate unique word count
            all_words = ' '.join(text_series).lower().split()
            unique_words = len(set(all_words))
            
            results[col] = {
                "basic_stats": {
                    "avg_words": float(word_counts.mean()),
                    "max_words": int(word_counts.max()),
                    "min_words": int(word_counts.min()),
                    "avg_chars": float(char_counts.mean()),
                    "max_chars": int(char_counts.max()),
                    "min_chars": int(char_counts.min())
                },
                "vocabulary_stats": {
                    "unique_words": unique_words,
                    "vocabulary_density": float(unique_words / len(all_words)),
                    "empty_ratio": float(text_series.str.strip().eq('').mean())
                },
                "common_words": pd.Series(all_words).value_counts().head(10).to_dict()
            }
            
        return results

    def _analyze_sentiment(self) -> Dict[str, Any]:
        """Analyze sentiment of text columns using basic lexicon-based approach."""
        text_cols = [col for col in self.categorical_cols if 
                    self.df[col].dtype == 'object' and 
                    self.df[col].str.len().mean() > 50]
        
        if not text_cols:
            return {}
            
        # Basic sentiment words (could be expanded)
        positive_words = {'good', 'great', 'excellent', 'best', 'amazing', 'wonderful', 'fantastic',
                         'happy', 'love', 'perfect', 'better', 'awesome', 'nice', 'positive'}
        negative_words = {'bad', 'worst', 'terrible', 'poor', 'awful', 'horrible', 'negative',
                         'hate', 'wrong', 'worse', 'disappointing', 'disappointed', 'useless'}
        
        results = {}
        for col in text_cols[:2]:  # Analyze top 2 text columns
            text_series = self.df[col].dropna().astype(str)
            
            # Convert to lowercase for analysis
            text_lower = text_series.str.lower()
            
            # Calculate sentiment scores
            positive_scores = text_lower.apply(lambda x: sum(word in x.split() for word in positive_words))
            negative_scores = text_lower.apply(lambda x: sum(word in x.split() for word in negative_words))
            
            results[col] = {
                "sentiment_distribution": {
                    "positive": float((positive_scores > negative_scores).mean()),
                    "negative": float((negative_scores > positive_scores).mean()),
                    "neutral": float((positive_scores == negative_scores).mean())
                },
                "average_scores": {
                    "positive_words": float(positive_scores.mean()),
                    "negative_words": float(negative_scores.mean())
                }
            }
            
        return results

    def _analyze_seasonality(self) -> Dict[str, Any]:
        """Analyze seasonality in time series data."""
        date_cols = [col for col in self.df.columns if 
                    self.df[col].dtype in ['datetime64[ns]', 'object'] and 
                    pd.to_datetime(self.df[col], errors='coerce').notna().mean() > 0.5]
        
        if not date_cols or not self.numeric_cols:
            return {}
            
        date_col = date_cols[0]
        df_temp = self.df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        
        results = {}
        for num_col in self.numeric_cols[:3]:  # Analyze top 3 numeric columns
            try:
                # Resample to daily frequency and interpolate
                series = df_temp.set_index(date_col)[num_col].resample('D').mean().interpolate()
                
                # Perform seasonal decomposition
                decomposition = sm.tsa.seasonal_decompose(
                    series,
                    period=min(30, len(series) // 2)  # Adjust period based on data length
                )
                
                # Calculate seasonality strength
                seasonal_strength = abs(decomposition.seasonal).mean() / abs(decomposition.resid).mean()
                
                # Find peaks and troughs in seasonal pattern
                seasonal = pd.Series(decomposition.seasonal)
                peaks = seasonal.nlargest(3)
                troughs = seasonal.nsmallest(3)
                
                # Calculate advanced seasonality metrics
                from scipy import stats
                acf = sm.tsa.stattools.acf(series, nlags=30)
                
                # Detect significant seasonal frequencies
                fft = np.fft.fft(series)
                freq = np.fft.fftfreq(len(series))
                significant_periods = []
                for idx in np.argsort(np.abs(fft))[-5:]:  # Top 5 frequencies
                    if freq[idx] > 0:  # Only positive frequencies
                        period = int(1 / freq[idx])
                        if 2 <= period <= len(series) // 3:  # Reasonable periods
                            significant_periods.append(period)
                
                results[num_col] = {
                    "seasonality_strength": float(seasonal_strength),
                    "has_seasonality": bool(seasonal_strength > 0.5),
                    "seasonal_peaks": {
                        str(idx.strftime('%Y-%m-%d')): float(val)
                        for idx, val in peaks.items()
                    },
                    "seasonal_troughs": {
                        str(idx.strftime('%Y-%m-%d')): float(val)
                        for idx, val in troughs.items()
                    },
                    "trend": {
                        "direction": "increasing" if decomposition.trend[-1] > decomposition.trend[0] else "decreasing",
                        "strength": float(abs(decomposition.trend[-1] - decomposition.trend[0]) / series.std())
                    },
                    "advanced_metrics": {
                        "autocorrelation": {
                            "lag_1": float(acf[1]),
                            "lag_7": float(acf[7]) if len(acf) > 7 else None,
                            "lag_30": float(acf[30]) if len(acf) > 30 else None
                        },
                        "significant_periods": significant_periods,
                        "seasonality_test": {
                            "statistic": float(stats.kruskal(*[series[i::12] for i in range(12)])[0])
                            if len(series) >= 24 else None,
                            "is_seasonal": bool(seasonal_strength > 0.5 and len(significant_periods) > 0)
                        }
                    }
                }
            except Exception as e:
                self.logger.warning(f"Could not analyze seasonality for {num_col}: {e}")
                continue
        
        return results

    def _analyze_trend(self) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        date_cols = [col for col in self.df.columns if 
                    self.df[col].dtype in ['datetime64[ns]', 'object'] and 
                    pd.to_datetime(self.df[col], errors='coerce').notna().mean() > 0.5]
        
        if not date_cols or not self.numeric_cols:
            return {}
            
        date_col = date_cols[0]
        df_temp = self.df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        
        results = {}
        for num_col in self.numeric_cols[:3]:  # Analyze top 3 numeric columns
            try:
                # Resample to daily frequency and interpolate
                series = df_temp.set_index(date_col)[num_col].resample('D').mean().interpolate()
                
                # Calculate linear trend
                x = np.arange(len(series))
                y = series.values
                z = np.polyfit(x, y, 1)
                slope = float(z[0])
                
                # Calculate growth metrics
                total_change = float(series.iloc[-1] - series.iloc[0])
                percent_change = float((series.iloc[-1] / series.iloc[0] - 1) * 100)
                
                # Calculate volatility and momentum
                volatility = float(series.std() / series.mean())  # Coefficient of variation
                momentum = float(series.diff().mean())  # Average daily change
                
                results[num_col] = {
                    "trend_metrics": {
                        "direction": "increasing" if slope > 0 else "decreasing",
                        "slope": slope,
                        "total_change": total_change,
                        "percent_change": percent_change
                    },
                    "volatility_metrics": {
                        "volatility": volatility,
                        "momentum": momentum,
                        "stability": "stable" if volatility < 0.1 else "volatile"
                    },
                    "change_points": {
                        "max_increase": float(series.diff().max()),
                        "max_decrease": float(series.diff().min()),
                        "significant_changes": len(series.diff()[abs(series.diff()) > 2 * series.diff().std()])
                    }
                }
            except Exception as e:
                self.logger.warning(f"Could not analyze trend for {num_col}: {e}")
                
        return results

    def _analyze_forecast_potential(self) -> Dict[str, Any]:
        """Analyze potential for forecasting in time series data."""
        date_cols = [col for col in self.df.columns if 
                    self.df[col].dtype in ['datetime64[ns]', 'object'] and 
                    pd.to_datetime(self.df[col], errors='coerce').notna().mean() > 0.5]
        
        if not date_cols or not self.numeric_cols:
            return {}
            
        date_col = date_cols[0]
        df_temp = self.df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        
        results = {}
        for num_col in self.numeric_cols[:3]:  # Analyze top 3 numeric columns
            try:
                # Resample to daily frequency and interpolate
                series = df_temp.set_index(date_col)[num_col].resample('D').mean().interpolate()
                
                # Check for stationarity
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(series.dropna())
                
                # Calculate autocorrelation
                autocorr = series.autocorr()
                
                # Calculate basic metrics
                results[num_col] = {
                    "data_quality": {
                        "data_points": len(series),
                        "missing_ratio": float(series.isna().mean()),
                        "frequency": "daily"
                    },
                    "stationarity": {
                        "is_stationary": bool(adf_result[1] < 0.05),
                        "p_value": float(adf_result[1]),
                        "needs_differencing": not (adf_result[1] < 0.05)
                    },
                    "autocorrelation": {
                        "lag_1": float(autocorr),
                        "has_strong_autocorr": abs(autocorr) > 0.7
                    },
                    "seasonality_info": {
                        "has_seasonality": bool("has_seasonality" in self.analysis_results.get("seasonality", {}).get(num_col, {})),
                        "seasonality_strength": float(self.analysis_results.get("seasonality", {}).get(num_col, {}).get("seasonality_strength", 0))
                    },
                    "recommendation": {
                        "suitable_for_forecasting": bool(
                            len(series) >= 30 and  # Enough data points
                            series.isna().mean() < 0.2 and  # Not too many missing values
                            abs(autocorr) > 0.7  # Strong autocorrelation
                        ),
                        "suggested_models": self._suggest_forecast_models(series)
                    }
                }
            except Exception as e:
                self.logger.warning(f"Could not analyze forecast potential for {num_col}: {e}")
                
        return results
        
    def _suggest_forecast_models(self, series: pd.Series) -> List[str]:
        """Suggest appropriate forecasting models based on data characteristics."""
        suggestions = []
        
        # Basic time series models
        suggestions.append("ARIMA")  # Always include ARIMA as baseline
        
        # Check for seasonality
        if hasattr(series, 'index') and len(series) >= 30:
            try:
                decomposition = sm.tsa.seasonal_decompose(
                    series,
                    period=min(30, len(series) // 2)
                )
                if abs(decomposition.seasonal).mean() / abs(decomposition.resid).mean() > 0.5:
                    suggestions.append("SARIMA")  # Add seasonal model
            except Exception:
                pass
        
        # Check for trend
        if len(series) >= 10:
            try:
                z = np.polyfit(range(len(series)), series.values, 1)
                if abs(z[0]) > 0.1 * series.std():  # Significant trend
                    suggestions.extend(["Prophet", "Exponential Smoothing"])
            except Exception:
                pass
        
        # Add advanced models if enough data
        if len(series) >= 100:
            suggestions.append("LSTM")
            if len(series) >= 1000:
                suggestions.append("Neural Prophet")
        
        return suggestions

    def _analyze_spatial_patterns(self) -> Dict[str, Any]:
        """Analyze spatial patterns in geographic data."""
        lat_cols = [col for col in self.numeric_cols if 
                   any(term in col.lower() for term in ['lat', 'latitude'])]
        long_cols = [col for col in self.numeric_cols if 
                    any(term in col.lower() for term in ['long', 'longitude'])]
        loc_cols = [col for col in self.categorical_cols if 
                   any(term in col.lower() for term in 
                       ['country', 'city', 'state', 'region', 'location'])]
        
        if not (lat_cols and long_cols) and not loc_cols:
            return {}
            
        results = {}
        
        if lat_cols and long_cols:
            lat_col, long_col = lat_cols[0], long_cols[0]
            coords_df = self.df[[lat_col, long_col]].dropna()
            
            results["coordinate_stats"] = {
                "center": {
                    "latitude": float(coords_df[lat_col].mean()),
                    "longitude": float(coords_df[long_col].mean())
                },
                "spread": {
                    "latitude_std": float(coords_df[lat_col].std()),
                    "longitude_std": float(coords_df[long_col].std())
                },
                "bounds": {
                    "north": float(coords_df[lat_col].max()),
                    "south": float(coords_df[lat_col].min()),
                    "east": float(coords_df[long_col].max()),
                    "west": float(coords_df[long_col].min())
                }
            }
            
        if loc_cols:
            for col in loc_cols:
                location_counts = self.df[col].value_counts()
                results[f"{col}_stats"] = {
                    "top_locations": location_counts.head(10).to_dict(),
                    "unique_locations": len(location_counts),
                    "concentration": float(location_counts.head(5).sum() / len(self.df))
                }
            
        return results

    def _analyze_regional_stats(self) -> Dict[str, Any]:
        """Analyze statistics by geographic regions."""
        loc_cols = [col for col in self.categorical_cols if 
                   any(term in col.lower() for term in 
                       ['country', 'city', 'state', 'region', 'location'])]
        
        if not loc_cols or not self.numeric_cols:
            return {}
            
        results = {}
        loc_col = loc_cols[0]  # Use first location column
        
        # Calculate regional statistics for each numeric column
        for num_col in self.numeric_cols[:3]:  # Analyze top 3 numeric columns
            regional_stats = self.df.groupby(loc_col)[num_col].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).sort_values('count', ascending=False).head(10)
            
            results[num_col] = {
                region: {
                    "mean": float(stats['mean']),
                    "std": float(stats['std']),
                    "min": float(stats['min']),
                    "max": float(stats['max']),
                    "count": int(stats['count'])
                }
                for region, stats in regional_stats.iterrows()
            }
            
        return results

    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance using various methods."""
        if len(self.numeric_cols) < 2:
            return {}
            
        # Prepare data
        X = self.df[self.numeric_cols].copy()
        X = X.fillna(X.mean())
        
        results = {}
        
        # Correlation-based importance
        corr_importance = abs(X.corr()).mean().sort_values(ascending=False)
        results["correlation_importance"] = {
            name: float(importance)
            for name, importance in corr_importance.head(10).items()
        }
        
        # Variance-based importance
        var_importance = X.var().sort_values(ascending=False)
        results["variance_importance"] = {
            name: float(importance)
            for name, importance in var_importance.head(10).items()
        }
        
        # Try Random Forest importance if possible
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
            rf.fit(X, X.iloc[:, 0])  # Use first column as target for demonstration
            rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
            results["random_forest_importance"] = {
                name: float(importance)
                for name, importance in rf_importance.sort_values(ascending=False).head(10).items()
            }
        except Exception as e:
            self.logger.warning(f"Could not calculate Random Forest importance: {e}")
        
        return results

    def _analyze_predictive_potential(self) -> Dict[str, Any]:
        """Analyze the potential for predictive modeling on the dataset."""
        results = {}
        
        # Check for classification potential
        categorical_targets = [col for col in self.categorical_cols if 
                             self.df[col].nunique() < 10 and 
                             'id' not in col.lower()]
        
        if categorical_targets:
            target = categorical_targets[0]
            class_dist = self.df[target].value_counts(normalize=True)
            
            results["classification"] = {
                "target_variable": target,
                "n_classes": len(class_dist),
                "class_balance": {
                    "ratio": float(class_dist.min() / class_dist.max()),
                    "interpretation": "balanced" if class_dist.min() / class_dist.max() > 0.3 else "imbalanced"
                },
                "sample_sufficiency": {
                    "samples_per_class": self.df[target].value_counts().to_dict(),
                    "sufficient": all(count >= 30 for count in self.df[target].value_counts())
                },
                "feature_readiness": {
                    "numeric_features": len(self.numeric_cols),
                    "categorical_features": len(self.categorical_cols),
                    "missing_value_impact": float(self.df[self.numeric_cols + [target]].isnull().mean().mean())
                }
            }
        
        # Check for regression potential
        numeric_targets = [col for col in self.numeric_cols if 
                         'id' not in col.lower() and
                         'index' not in col.lower() and
                         self.df[col].nunique() > 10]
        
        if numeric_targets:
            target = numeric_targets[0]
            target_series = self.df[target].dropna()
            
            from scipy import stats
            
            # Calculate skewness and normality
            skewness = float(target_series.skew())
            _, normality_p_value = stats.normaltest(target_series)
            
            # Calculate potential feature relationships
            correlations = abs(self.df[self.numeric_cols].corr()[target]).sort_values(ascending=False)
            strong_predictors = correlations[correlations > 0.3].index.tolist()
            
            results["regression"] = {
                "target_variable": target,
                "distribution": {
                    "skewness": skewness,
                    "is_normal": float(normality_p_value) > 0.05,
                    "range": {
                        "min": float(target_series.min()),
                        "max": float(target_series.max()),
                        "std": float(target_series.std())
                    }
                },
                "feature_relationships": {
                    "strong_predictors": strong_predictors,
                    "max_correlation": float(correlations.iloc[1] if len(correlations) > 1 else 0),
                    "n_strong_predictors": len(strong_predictors) - 1  # Exclude target itself
                },
                "data_quality": {
                    "missing_values": float(self.df[target].isnull().mean()),
                    "unique_ratio": float(self.df[target].nunique() / len(self.df)),
                    "outlier_ratio": float(len(target_series[abs(stats.zscore(target_series)) > 3]) / len(target_series))
                }
            }
            
        # General predictive potential assessment
        if results:
            feature_completeness = 1 - self.df[self.numeric_cols].isnull().mean().mean()
            n_samples = len(self.df)
            n_features = len(self.numeric_cols)
            
            results["overall_assessment"] = {
                "data_volume": {
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "samples_per_feature": float(n_samples / max(1, n_features)),
                    "sufficient_volume": n_samples >= 100 and n_samples >= 10 * n_features
                },
                "data_quality": {
                    "feature_completeness": float(feature_completeness),
                    "quality_score": float(feature_completeness * min(1, n_samples / (10 * n_features)))
                },
                "recommendation": {
                    "suitable_for_ml": n_samples >= 100 and feature_completeness > 0.7,
                    "suggested_approaches": self._suggest_modeling_approaches(results)
                }
            }
            
        return results
        
    def _suggest_modeling_approaches(self, predictive_results: Dict[str, Any]) -> List[str]:
        """Suggest appropriate modeling approaches based on the predictive analysis results."""
        suggestions = []
        
        if "classification" in predictive_results:
            class_info = predictive_results["classification"]
            
            # Binary vs multiclass
            if class_info["n_classes"] == 2:
                suggestions.extend([
                    "logistic_regression",
                    "random_forest_classifier",
                    "gradient_boosting_classifier"
                ])
            else:
                suggestions.extend([
                    "random_forest_classifier",
                    "gradient_boosting_classifier",
                    "neural_network_classifier"
                ])
                
            # Handle imbalanced data
            if class_info["class_balance"]["ratio"] < 0.3:
                suggestions.extend([
                    "smote_balancing",
                    "weighted_models"
                ])
                
        if "regression" in predictive_results:
            reg_info = predictive_results["regression"]
            
            # Basic suggestions
            suggestions.extend([
                "random_forest_regressor",
                "gradient_boosting_regressor"
            ])
            
            # Handle non-normal distribution
            if not reg_info["distribution"]["is_normal"]:
                suggestions.extend([
                    "target_transformation",
                    "quantile_regression"
                ])
                
            # Handle high feature relationships
            if reg_info["feature_relationships"]["n_strong_predictors"] > 5:
                suggestions.extend([
                    "feature_selection",
                    "regularized_regression"
                ])
                
        return list(set(suggestions))  # Remove duplicates

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_file>")
        sys.exit(1)
        
    # Get absolute paths
    csv_path = os.path.abspath(sys.argv[1])
    
    # Validate input file
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} does not exist")
        sys.exit(1)
    
    if not csv_path.lower().endswith('.csv'):
        print("Error: Input file must be a CSV file")
        sys.exit(1)
        
    # Create output directory based on dataset name
    dataset_name = Path(csv_path).stem
    current_dir = Path.cwd()
    output_dir = current_dir / dataset_name
    output_dir.mkdir(exist_ok=True)
    
    # Change to output directory
    os.chdir(output_dir)
    
    # Create and run analyzer
    analyzer = DataAnalyzer(csv_path)
    
    # Run each step safely
    analyzer.safe_analyze()
    analyzer.safe_visualize()
    analyzer.safe_generate_story()
    
    # Log completion status
    if analyzer.errors:
        logger.warning(f"Analysis completed with {len(analyzer.errors)} issues. Check README.md for details.")
    else:
        logger.info(f"Analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main() 