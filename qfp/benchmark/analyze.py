# analyze_benchmark.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_benchmark_results(results_file="/home/luiz/repositories/qfp/qfp/benchmark/data/qfp_benchmark_results.json"):
    """Analisa detalhadamente os resultados do benchmark"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Criar DataFrame com resultados das queries
    queries_data = []
    for query_name, query_result in results['queries'].items():
        query_data = {
            'query': query_name,
            'precision': query_result['precision'],
            'recall': query_result['recall'],
            'f1_score': query_result['f1_score'],
            'extraction_time': query_result['extraction_time'],
            'total_search_time': query_result['total_search_time'],
            'total_matches': query_result['total_matches'],
            'true_positives': len(query_result['true_positives']),
            'false_positives': len(query_result['false_positives']),
            'false_negatives': len(query_result['false_negatives'])
        }
        queries_data.append(query_data)
    
    df = pd.DataFrame(queries_data)
    
    # Configurar estilo dos plots
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análise Detalhada do Benchmark - Sistema QFP', fontsize=16, fontweight='bold')
    
    # 1. Distribuição de Precision, Recall e F1-Score
    metrics_to_plot = ['precision', 'recall', 'f1_score']
    for i, metric in enumerate(metrics_to_plot):
        axes[0, i].hist(df[metric], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, i].axvline(df[metric].mean(), color='red', linestyle='--', label=f'Média: {df[metric].mean():.3f}')
        axes[0, i].set_xlabel(metric.capitalize())
        axes[0, i].set_ylabel('Frequência')
        axes[0, i].set_title(f'Distribuição de {metric.capitalize()}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    # 2. Distribuição de Tempos
    time_metrics = ['extraction_time', 'total_search_time']
    colors = ['lightcoral', 'lightgreen']
    for i, metric in enumerate(time_metrics):
        axes[1, i].hist(df[metric], bins=20, alpha=0.7, color=colors[i], edgecolor='black')
        axes[1, i].axvline(df[metric].mean(), color='red', linestyle='--', label=f'Média: {df[metric].mean():.2f}s')
        axes[1, i].set_xlabel('Tempo (segundos)')
        axes[1, i].set_ylabel('Frequência')
        axes[1, i].set_title(f'Distribuição de {metric.replace("_", " ").title()}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    # 3. Scatter Plot: Tempo de Busca vs F1-Score
    scatter = axes[1, 2].scatter(df['total_search_time'], df['f1_score'], 
                                c=df['total_matches'], cmap='viridis', alpha=0.6)
    axes[1, 2].set_xlabel('Tempo Total de Busca (s)')
    axes[1, 2].set_ylabel('F1-Score')
    axes[1, 2].set_title('Relação: Tempo de Busca vs F1-Score')
    axes[1, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 2], label='Número de Matches')
    
    plt.tight_layout()
    plt.savefig('benchmark_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estatísticas detalhadas
    print("="*60)
    print("ANÁLISE ESTATÍSTICA DETALHADA")
    print("="*60)
    
    print("\n--- ESTATÍSTICAS DAS MÉTRICAS PRINCIPAIS ---")
    print(df[['precision', 'recall', 'f1_score']].describe())
    
    print("\n--- ESTATÍSTICAS DE TEMPO ---")
    print(df[['extraction_time', 'total_search_time']].describe())
    
    print("\n--- ANÁLISE DE PERFORMANCE POR QUERY ---")
    print("\nTop 10 queries com melhor F1-Score:")
    top_f1 = df.nlargest(10, 'f1_score')[['query', 'f1_score', 'precision', 'recall']]
    print(top_f1.to_string(index=False))
    
    print("\nTop 10 queries mais rápidas:")
    top_fast = df.nsmallest(10, 'total_search_time')[['query', 'total_search_time', 'f1_score']]
    print(top_fast.to_string(index=False))
    
    # Análise de componentes de tempo
    print("\n--- ANÁLISE DE COMPONENTES DE TEMPO ---")
    time_components = []
    for query_name, query_result in results['queries'].items():
        time_components.append({
            'query': query_name,
            'faiss_search': query_result['faiss_search_time'],
            'filter': query_result['filter_time'],
            'histogram': query_result['process_histogram_time'],
            'matches': query_result['matches_time']
        })
    
    time_df = pd.DataFrame(time_components)
    print("\nTempo médio por componente:")
    print(time_df[['faiss_search', 'filter', 'histogram', 'matches']].mean())
    
    # Salvar análise detalhada em CSV
    df.to_csv('detailed_benchmark_analysis.csv', index=False)
    print(f"\nAnálise detalhada salva em: detailed_benchmark_analysis.csv")

if __name__ == "__main__":
    analyze_benchmark_results()