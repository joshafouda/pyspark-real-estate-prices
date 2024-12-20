from pyspark.sql import SparkSession

def create_spark_session(app_name="RealEstate_ML"):
    """
    Crée une session Spark optimisée pour une machine avec :
    - RAM : 32GB
    - CPU : Intel Core i7-1255U (12 coeurs)
    - Stockage : 1TB
    
    Parameters:
        app_name (str): Nom de l'application Spark
        
    Returns:
        SparkSession: Session Spark configurée
    """
    # Calculer les ressources optimales
    # Utiliser 80% de la RAM disponible pour Spark
    executor_memory = "25g"  # 80% de 32GB
    
    # Réserver de la mémoire pour le overhead et la gestion des données
    memory_overhead = "5g"
    
    # Utiliser 80% des coeurs disponibles pour les tâches parallèles
    executor_cores = 10  # 80% des 12 coeurs
    
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", executor_memory) \
        .config("spark.driver.memoryOverhead", memory_overhead) \
        .config("spark.driver.cores", executor_cores) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
        .config("spark.sql.inMemoryColumnarStorage.batchSize", 10000) \
        .config("spark.sql.shuffle.partitions", executor_cores * 4) \
        .config("spark.default.parallelism", executor_cores * 4) \
        .config("spark.memory.fraction", 0.8) \
        .config("spark.memory.storageFraction", 0.3) \
        .config("spark.sql.autoBroadcastJoinThreshold", "100MB") \
        .config("spark.sql.broadcastTimeout", 300) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.maxResultSize", "5g") \
        .config("spark.kryoserializer.buffer.max", "1g") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+UseCompressedOops") \
        .getOrCreate()
