from sequence_embeddings import RouletteGraphDB, RouletteEmbedding, SequenceDatabase
import time

if __name__ == "__main__":
    print("\n🎲 INICIANDO SISTEMA DE EMBEDDINGS DE RULETA 🎲")
    start_total = time.time()
    
    try:
        # 1. Inicializar conexión a Neo4j
        graph_db = RouletteGraphDB(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="123456789"
        )
        
        # 2. Cargar modelo y encoders
        print("\n=== Cargando Recursos ===")
        from tensorflow.keras.models import load_model
        import pickle
        
        print("1. Cargando modelo...")
        modelo = load_model("modelo_ruleta_CMulti_Cursor2.h5")
        print("✓ Modelo cargado correctamente")
        
        print("\n2. Cargando encoders...")
        with open('encoders.pickle', 'rb') as handle:
            encoders = pickle.load(handle)
        print("✓ Encoders cargados correctamente")
        
        # 3. Crear sistema de embeddings
        embedding_system = RouletteEmbedding(
            model=modelo,
            csv_path="secuencia_ruleta_V2.csv",
            encoders=encoders,
            sequence_length=50,  # Longer sequences
            overlap_factor=0.7,  # More overlap
            batch_size=64       # Larger batches
        )
        
        # 4. Obtener y probar una secuencia
        print("\n=== Probando Sistema ===")
        ultima_secuencia = embedding_system.get_historical_sequence()
        print(f"1. Última secuencia obtenida (últimos 5 números): ...{ultima_secuencia[-5:]}")
        
        embedding = embedding_system.get_sequence_embedding(ultima_secuencia)
        print(f"2. Embedding generado con forma: {embedding.shape if embedding is not None else 'None'}")
        
        # 5. Mostrar estadísticas actualizadas
        print("\n=== Estadísticas de la Base de Datos ===")
        stats = graph_db.get_database_stats()
        print(f"• Secuencias almacenadas: {stats['sequences']}")
        print(f"• Relaciones creadas: {stats['relationships']}")
        
        # 6. Mostrar tiempo total
        tiempo_total = time.time() - start_total
        print(f"\n✅ SISTEMA INICIALIZADO CORRECTAMENTE")
        print(f"⏱️  Tiempo total de ejecución: {tiempo_total:.2f} segundos")
        
        # 7. Validar utilidad de la base de datos
        print("\n=== Validando Utilidad de la Base de Datos ===")
        # Obtener algunas secuencias de prueba
        test_sequences = [embedding_system.get_historical_sequence()]  # Obtener una única secuencia
        
        # Check if the method exists before calling it
        if hasattr(embedding_system, 'validate_database_utility'):
            validation_results = embedding_system.validate_database_utility(test_sequences)
        else:
            print("❌ ERROR: 'validate_database_utility' method does not exist in 'RouletteEmbedding' class.")
            # Optionally, implement a placeholder or alternative validation logic here
            validation_results = None  # Placeholder for validation results
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise 
    