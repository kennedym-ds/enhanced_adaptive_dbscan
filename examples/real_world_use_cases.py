"""
Real-World Use Cases for Enhanced Adaptive DBSCAN

This example demonstrates practical applications of the Enhanced Adaptive DBSCAN
algorithm in various real-world scenarios including anomaly detection, customer
segmentation, and sensor data analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN


def anomaly_detection_example():
    """
    Demonstrate anomaly detection in network traffic data.
    
    This example shows how to use Enhanced Adaptive DBSCAN for identifying
    anomalous patterns in network traffic data.
    """
    print("=" * 60)
    print("ANOMALY DETECTION IN NETWORK TRAFFIC")
    print("=" * 60)
    
    # Generate synthetic network traffic data
    np.random.seed(42)
    n_samples = 1000
    
    # Normal traffic patterns
    normal_traffic = {
        'packet_size': np.random.normal(1500, 200, int(n_samples * 0.9)),
        'request_rate': np.random.exponential(10, int(n_samples * 0.9)),
        'response_time': np.random.lognormal(1, 0.5, int(n_samples * 0.9)),
        'connection_count': np.random.poisson(5, int(n_samples * 0.9))
    }
    
    # Anomalous traffic patterns (DDoS attack simulation)
    anomalous_traffic = {
        'packet_size': np.random.normal(64, 10, int(n_samples * 0.1)),  # Small packets
        'request_rate': np.random.exponential(100, int(n_samples * 0.1)),  # High rate
        'response_time': np.random.exponential(5, int(n_samples * 0.1)),  # Fast responses
        'connection_count': np.random.poisson(50, int(n_samples * 0.1))  # Many connections
    }
    
    # Combine data
    traffic_data = pd.DataFrame({
        'packet_size': np.concatenate([normal_traffic['packet_size'], 
                                     anomalous_traffic['packet_size']]),
        'request_rate': np.concatenate([normal_traffic['request_rate'], 
                                      anomalous_traffic['request_rate']]),
        'response_time': np.concatenate([normal_traffic['response_time'], 
                                       anomalous_traffic['response_time']]),
        'connection_count': np.concatenate([normal_traffic['connection_count'], 
                                          anomalous_traffic['connection_count']])
    })
    
    # Add timestamps
    start_time = datetime.now() - timedelta(hours=1)
    traffic_data['timestamp'] = [start_time + timedelta(seconds=i*3.6) 
                               for i in range(len(traffic_data))]
    
    # Create ground truth labels (0=normal, 1=anomaly)
    true_labels = np.concatenate([np.zeros(int(n_samples * 0.9)), 
                                np.ones(int(n_samples * 0.1))])
    
    print(f"Dataset created: {len(traffic_data)} traffic records")
    print(f"Normal traffic: {sum(true_labels == 0)} records")
    print(f"Anomalous traffic: {sum(true_labels == 1)} records")
    
    # Prepare features for clustering
    feature_columns = ['packet_size', 'request_rate', 'response_time', 'connection_count']
    X = traffic_data[feature_columns].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Enhanced Adaptive DBSCAN for anomaly detection
    anomaly_detector = EnhancedAdaptiveDBSCAN(
        eps=0.5,
        min_samples=10,
        enable_mdbscan=True,
        enable_boundary_refinement=True,
        adaptive_eps=True,
        enable_stability_analysis=True,
        random_state=42
    )
    
    print(f"\nRunning anomaly detection...")
    anomaly_detector.fit(X_scaled)
    cluster_labels = anomaly_detector.labels_
    
    # Identify anomalies (points in small clusters or noise)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    anomaly_threshold = 50  # Clusters with fewer than 50 points are considered anomalous
    
    anomalous_clusters = set()
    for label, count in zip(unique_labels, counts):
        if count < anomaly_threshold:
            anomalous_clusters.add(label)
    
    # Create binary anomaly predictions
    anomaly_predictions = np.array([1 if label in anomalous_clusters else 0 
                                  for label in cluster_labels])
    
    # Calculate detection metrics
    true_positives = sum((true_labels == 1) & (anomaly_predictions == 1))
    false_positives = sum((true_labels == 0) & (anomaly_predictions == 1))
    true_negatives = sum((true_labels == 0) & (anomaly_predictions == 0))
    false_negatives = sum((true_labels == 1) & (anomaly_predictions == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAnomaly Detection Results:")
    print(f"  Clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    print(f"  Anomalous clusters: {len(anomalous_clusters)}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1_score:.3f}")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  True Negatives: {true_negatives}")
    print(f"  False Negatives: {false_negatives}")
    
    return traffic_data, cluster_labels, anomaly_predictions, true_labels


def customer_segmentation_example():
    """
    Demonstrate customer segmentation for e-commerce data.
    
    This example shows how to segment customers based on their behavior
    and transaction patterns.
    """
    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION FOR E-COMMERCE")
    print("=" * 60)
    
    # Generate synthetic customer data
    np.random.seed(42)
    n_customers = 2000
    
    # Customer segments with different behaviors
    segments = {
        'High Value': {
            'size': int(n_customers * 0.15),
            'avg_order_value': (200, 50),
            'purchase_frequency': (15, 5),
            'session_duration': (45, 15),
            'support_tickets': (2, 1)
        },
        'Regular': {
            'size': int(n_customers * 0.6),
            'avg_order_value': (80, 20),
            'purchase_frequency': (6, 2),
            'session_duration': (20, 8),
            'support_tickets': (4, 2)
        },
        'Occasional': {
            'size': int(n_customers * 0.2),
            'avg_order_value': (45, 15),
            'purchase_frequency': (2, 1),
            'session_duration': (10, 5),
            'support_tickets': (1, 1)
        },
        'At Risk': {
            'size': int(n_customers * 0.05),
            'avg_order_value': (25, 10),
            'purchase_frequency': (1, 0.5),
            'session_duration': (5, 2),
            'support_tickets': (8, 3)
        }
    }
    
    customer_data = []
    true_segments = []
    
    for segment_name, params in segments.items():
        for _ in range(params['size']):
            customer = {
                'customer_id': len(customer_data) + 1,
                'avg_order_value': max(0, np.random.normal(*params['avg_order_value'])),
                'purchase_frequency': max(0, np.random.normal(*params['purchase_frequency'])),
                'session_duration': max(0, np.random.normal(*params['session_duration'])),
                'support_tickets': max(0, int(np.random.normal(*params['support_tickets']))),
                'days_since_last_purchase': np.random.exponential(30),
                'total_spent': 0,  # Will calculate
                'loyalty_score': np.random.uniform(0, 1)
            }
            
            # Calculate total spent
            customer['total_spent'] = customer['avg_order_value'] * customer['purchase_frequency']
            
            customer_data.append(customer)
            true_segments.append(segment_name)
    
    customer_df = pd.DataFrame(customer_data)
    
    print(f"Customer dataset created: {len(customer_df)} customers")
    for segment in segments.keys():
        count = true_segments.count(segment)
        print(f"  {segment}: {count} customers ({count/len(customer_df)*100:.1f}%)")
    
    # Prepare features for clustering
    feature_columns = ['avg_order_value', 'purchase_frequency', 'session_duration', 
                      'support_tickets', 'days_since_last_purchase', 'total_spent', 'loyalty_score']
    X = customer_df[feature_columns].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Enhanced Adaptive DBSCAN for customer segmentation
    segmentation_model = EnhancedAdaptiveDBSCAN(
        eps=0.6,
        min_samples=15,
        enable_mdbscan=True,
        enable_hierarchical_clustering=True,
        enable_quality_analysis=True,
        adaptive_eps=True,
        density_scaling=1.2,
        random_state=42
    )
    
    print(f"\nRunning customer segmentation...")
    segmentation_model.fit(X_scaled)
    cluster_labels = segmentation_model.labels_
    
    # Analyze results
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = list(cluster_labels).count(-1)
    
    print(f"\nSegmentation Results:")
    print(f"  Segments identified: {n_clusters}")
    print(f"  Outlier customers: {n_outliers}")
    
    if n_clusters > 1:
        silhouette = silhouette_score(X_scaled, cluster_labels)
        print(f"  Silhouette score: {silhouette:.3f}")
    
    # Analyze each segment
    customer_df['cluster'] = cluster_labels
    
    print(f"\nSegment Characteristics:")
    print(f"{'Cluster':<8} {'Size':<6} {'Avg Order':<12} {'Frequency':<11} {'Total Spent':<12} {'Loyalty':<8}")
    print("-" * 65)
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            cluster_name = "Outliers"
        else:
            cluster_name = f"Segment {cluster_id}"
        
        cluster_data = customer_df[customer_df['cluster'] == cluster_id]
        
        avg_order = cluster_data['avg_order_value'].mean()
        avg_freq = cluster_data['purchase_frequency'].mean()
        avg_spent = cluster_data['total_spent'].mean()
        avg_loyalty = cluster_data['loyalty_score'].mean()
        
        print(f"{cluster_name:<8} {len(cluster_data):<6} ${avg_order:<11.2f} "
              f"{avg_freq:<11.1f} ${avg_spent:<11.2f} {avg_loyalty:<8.3f}")
    
    # Business insights
    print(f"\nBusiness Insights:")
    
    # Identify high-value segments
    cluster_spending = customer_df.groupby('cluster')['total_spent'].mean()
    if len(cluster_spending) > 0:
        highest_value_cluster = cluster_spending.idxmax()
        print(f"  Highest value segment: Cluster {highest_value_cluster} "
              f"(${cluster_spending[highest_value_cluster]:.2f} avg spending)")
    
    # Identify at-risk customers
    at_risk_customers = customer_df[
        (customer_df['days_since_last_purchase'] > 60) & 
        (customer_df['support_tickets'] > 5)
    ]
    print(f"  At-risk customers identified: {len(at_risk_customers)}")
    
    return customer_df, cluster_labels, segmentation_model


def sensor_data_analysis_example():
    """
    Demonstrate clustering of IoT sensor data for pattern discovery.
    
    This example shows how to analyze time-series sensor data to identify
    operational patterns and anomalies.
    """
    print("\n" + "=" * 60)
    print("IOT SENSOR DATA PATTERN ANALYSIS")
    print("=" * 60)
    
    # Generate synthetic IoT sensor data
    np.random.seed(42)
    n_readings = 2000
    
    # Simulate different operational modes
    timestamps = pd.date_range(start='2024-01-01', periods=n_readings, freq='H')
    
    sensor_data = []
    true_modes = []
    
    for i, timestamp in enumerate(timestamps):
        # Simulate different operational patterns based on time
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        if 6 <= hour <= 22 and day_of_week < 5:  # Working hours, weekdays
            mode = 'normal_operation'
            temperature = np.random.normal(75, 5)
            humidity = np.random.normal(45, 8)
            pressure = np.random.normal(1013, 15)
            vibration = np.random.normal(2, 0.5)
            power_consumption = np.random.normal(120, 20)
            
        elif 22 < hour or hour < 6 or day_of_week >= 5:  # Off hours or weekends
            mode = 'low_activity'
            temperature = np.random.normal(70, 3)
            humidity = np.random.normal(40, 5)
            pressure = np.random.normal(1013, 10)
            vibration = np.random.normal(0.5, 0.2)
            power_consumption = np.random.normal(30, 10)
            
        else:
            mode = 'normal_operation'
            temperature = np.random.normal(75, 5)
            humidity = np.random.normal(45, 8)
            pressure = np.random.normal(1013, 15)
            vibration = np.random.normal(2, 0.5)
            power_consumption = np.random.normal(120, 20)
        
        # Add some maintenance events
        if np.random.random() < 0.05:  # 5% chance of maintenance
            mode = 'maintenance'
            temperature = np.random.normal(80, 10)
            humidity = np.random.normal(50, 15)
            pressure = np.random.normal(1010, 20)
            vibration = np.random.normal(5, 2)
            power_consumption = np.random.normal(200, 50)
        
        # Add some anomalies
        if np.random.random() < 0.02:  # 2% chance of anomaly
            mode = 'anomaly'
            temperature = np.random.normal(95, 15)  # High temperature
            humidity = np.random.normal(80, 20)     # High humidity
            pressure = np.random.normal(980, 30)    # Low pressure
            vibration = np.random.normal(8, 3)      # High vibration
            power_consumption = np.random.normal(300, 100)  # High power
        
        sensor_reading = {
            'timestamp': timestamp,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'vibration': vibration,
            'power_consumption': power_consumption,
            'hour': hour,
            'day_of_week': day_of_week
        }
        
        sensor_data.append(sensor_reading)
        true_modes.append(mode)
    
    sensor_df = pd.DataFrame(sensor_data)
    
    print(f"Sensor dataset created: {len(sensor_df)} readings over {n_readings/24:.1f} days")
    mode_counts = pd.Series(true_modes).value_counts()
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count} readings ({count/len(sensor_df)*100:.1f}%)")
    
    # Engineer features for clustering
    # Include rolling statistics to capture temporal patterns
    window_size = 24  # 24-hour window
    
    sensor_df['temp_rolling_mean'] = sensor_df['temperature'].rolling(window=window_size, min_periods=1).mean()
    sensor_df['temp_rolling_std'] = sensor_df['temperature'].rolling(window=window_size, min_periods=1).std()
    sensor_df['vibration_rolling_max'] = sensor_df['vibration'].rolling(window=window_size, min_periods=1).max()
    sensor_df['power_rolling_mean'] = sensor_df['power_consumption'].rolling(window=window_size, min_periods=1).mean()
    
    # Select features for clustering
    feature_columns = [
        'temperature', 'humidity', 'pressure', 'vibration', 'power_consumption',
        'temp_rolling_mean', 'temp_rolling_std', 'vibration_rolling_max', 'power_rolling_mean',
        'hour', 'day_of_week'
    ]
    
    X = sensor_df[feature_columns].fillna(0).values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Enhanced Adaptive DBSCAN for pattern discovery
    pattern_analyzer = EnhancedAdaptiveDBSCAN(
        eps=0.8,
        min_samples=20,
        enable_mdbscan=True,
        enable_hierarchical_clustering=True,
        enable_quality_analysis=True,
        enable_boundary_refinement=True,
        adaptive_eps=True,
        density_scaling=1.5,
        random_state=42
    )
    
    print(f"\nRunning pattern analysis...")
    pattern_analyzer.fit(X_scaled)
    cluster_labels = pattern_analyzer.labels_
    
    # Analyze results
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_anomalies = list(cluster_labels).count(-1)
    
    print(f"\nPattern Analysis Results:")
    print(f"  Operational patterns identified: {n_clusters}")
    print(f"  Anomalous readings: {n_anomalies}")
    
    if n_clusters > 1:
        silhouette = silhouette_score(X_scaled, cluster_labels)
        print(f"  Silhouette score: {silhouette:.3f}")
    
    # Analyze each pattern
    sensor_df['pattern'] = cluster_labels
    
    print(f"\nPattern Characteristics:")
    print(f"{'Pattern':<8} {'Size':<6} {'Avg Temp':<10} {'Avg Hum':<10} {'Avg Vib':<10} {'Avg Power':<11}")
    print("-" * 65)
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            pattern_name = "Anomalies"
        else:
            pattern_name = f"Pattern {cluster_id}"
        
        cluster_data = sensor_df[sensor_df['pattern'] == cluster_id]
        
        avg_temp = cluster_data['temperature'].mean()
        avg_hum = cluster_data['humidity'].mean()
        avg_vib = cluster_data['vibration'].mean()
        avg_power = cluster_data['power_consumption'].mean()
        
        print(f"{pattern_name:<8} {len(cluster_data):<6} {avg_temp:<10.1f} "
              f"{avg_hum:<10.1f} {avg_vib:<10.2f} {avg_power:<11.1f}")
    
    # Operational insights
    print(f"\nOperational Insights:")
    
    # Identify peak consumption patterns
    power_by_pattern = sensor_df.groupby('pattern')['power_consumption'].mean()
    if len(power_by_pattern) > 0:
        highest_power_pattern = power_by_pattern.idxmax()
        print(f"  Highest power consumption pattern: Pattern {highest_power_pattern} "
              f"({power_by_pattern[highest_power_pattern]:.1f}W avg)")
    
    # Identify most stable patterns
    stability_by_pattern = sensor_df.groupby('pattern')['temp_rolling_std'].mean()
    if len(stability_by_pattern) > 0:
        most_stable_pattern = stability_by_pattern.idxmin()
        print(f"  Most stable operational pattern: Pattern {most_stable_pattern} "
              f"(std: {stability_by_pattern[most_stable_pattern]:.2f})")
    
    # Time-based pattern analysis
    pattern_by_hour = sensor_df.groupby(['hour', 'pattern']).size().unstack(fill_value=0)
    print(f"  Patterns show clear time-based variations: {len(pattern_by_hour.columns) > 1}")
    
    return sensor_df, cluster_labels, pattern_analyzer


def main():
    """Run all real-world use case examples."""
    print("Enhanced Adaptive DBSCAN - Real-World Use Cases")
    print("=" * 60)
    
    try:
        # Run all use case examples
        traffic_data, traffic_clusters, anomaly_preds, true_labels = anomaly_detection_example()
        customer_data, customer_clusters, segmentation_model = customer_segmentation_example()
        sensor_data, sensor_clusters, pattern_analyzer = sensor_data_analysis_example()
        
        print(f"\n" + "=" * 60)
        print("ALL USE CASE EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Summary insights
        print(f"\nSummary of Applications:")
        print(f"  1. Network Anomaly Detection: Identified {sum(anomaly_preds)} anomalous events")
        print(f"  2. Customer Segmentation: Found {len(set(customer_clusters))} customer segments")
        print(f"  3. IoT Pattern Analysis: Discovered {len(set(sensor_clusters))} operational patterns")
        
        return {
            'anomaly_detection': (traffic_data, traffic_clusters, anomaly_preds, true_labels),
            'customer_segmentation': (customer_data, customer_clusters, segmentation_model),
            'sensor_analysis': (sensor_data, sensor_clusters, pattern_analyzer)
        }
        
    except Exception as e:
        print(f"\nError occurred during examples: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\nUse case results available for further analysis.")
        print(f"Each example demonstrates different capabilities:")
        print(f"  - Anomaly detection with precision/recall metrics")
        print(f"  - Customer segmentation with business insights")
        print(f"  - Sensor pattern analysis with temporal features")
