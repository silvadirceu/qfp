# benchmark_faiss_simple.py
import json
import pandas as pd
import numpy as np
import time
import os
import glob
import random
import argparse
from collections import defaultdict
from qfp.db_mem import InMemoryQfpDB
from qfp import QueryFingerprint

class FAISSBenchmark:
    def __init__(self, ground_truth_csv, references_dir, queries_pickle_dir):
        """
        Inicializa o benchmark simplificado do FAISS
        
        Args:
            ground_truth_csv (str): Caminho para o arquivo CSV com anotações
            references_dir (str): Diretório com fingerprints de referência (.pkl)
            queries_pickle_dir (str): Diretório com fingerprints de query pré-processadas (.pkl)
        """
        self.ground_truth_csv = ground_truth_csv
        self.references_dir = references_dir
        self.queries_pickle_dir = queries_pickle_dir
        self.db = InMemoryQfpDB()
        self.ground_truth = None
        self.results = {}
        
        # Carregar gabarito
        self._load_ground_truth()
    
    def _load_ground_truth(self):
        """Carrega e processa o arquivo de ground truth"""
        df = pd.read_csv(self.ground_truth_csv)
        
        # Normaliza nomes dos arquivos
        df['query_file'] = df['query'].str.replace('.wav', '')
        df['reference_file'] = df['reference'].str.replace('.wav', '')
        
        # Cria dict: query -> set(referências corretas)
        self.ground_truth = (
            df.groupby('query_file')['reference_file']
            .apply(set)
            .to_dict()
        )
        
        print(f"Ground truth carregado: {len(self.ground_truth)} queries")
    
    def build_reference_database(self):
        """Constrói o banco de dados de referência e retorna tempo de carga"""
        print("Construindo banco de dados de referência...")
        start_time = time.time()
        
        self.db.store_all_pickles_from_directory(self.references_dir)
        
        load_time = time.time() - start_time
        print(f"Banco construído em {load_time:.2f} segundos")
        print(f"Total de referências: {len(self.db.fingerprints)}")
        print(f"Total de hashes no FAISS: {self.db.faiss_index.ntotal}")
        
        return load_time
    
    def load_query_fingerprints(self, max_queries=None, random_seed=None):
        """Carrega fingerprints de query pré-processadas"""
        if random_seed is not None:
            random.seed(random_seed)
        
        # Listar todos os arquivos pickle de query
        pickle_files = []
        for ext in ['*.pkl', '*.pickle']:
            pickle_files.extend(glob.glob(os.path.join(self.queries_pickle_dir, ext)))
        
        if not pickle_files:
            raise ValueError(f"Nenhum arquivo pickle encontrado em {self.queries_pickle_dir}")
        
        print(f"Encontrados {len(pickle_files)} arquivos de query pickle")
        
        # # Selecionar queries baseado no max_queries
        if max_queries and max_queries < len(pickle_files):
            selected_files = random.sample(pickle_files, max_queries)
        else:
            selected_files = pickle_files
        
        queries = {}
        loaded_count = 0
        
        for pickle_file in selected_files:
            try:
                # Extrair nome da query do arquivo
                filename = os.path.basename(pickle_file)
                query_name = filename.replace('_query_fingerprint.pkl', '').replace('.pkl', '')
                
                # Verificar se a query está no ground truth
                if query_name not in self.ground_truth:
                    print(f"Query {query_name} não encontrada no ground truth, pulando...")
                    continue
                
                # Carregar fingerprint
                query_fp = QueryFingerprint.load_from_pickle(pickle_file)
                queries[query_name] = query_fp
                loaded_count += 1
                
            except Exception as e:
                print(f"Erro ao carregar {pickle_file}: {e}")
        
        print(f"Carregadas {loaded_count} queries")
        return queries
    
    def _save_partial_results(self, results, output_dir, e_radius, checkpoint_info=None):
        """Salva resultados parciais com informações do checkpoint"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Nome do arquivo inclui o raio e indica que é parcial
        filename = f"faiss_benchmark_radius_{e_radius}_partial.json".replace('.', '_')
        output_path = os.path.join(output_dir, filename)
        
        # Adicionar informações do checkpoint
        if checkpoint_info:
            results['checkpoint'] = checkpoint_info
        
        results['metadata']['last_update'] = time.strftime("%Y-%m-%d %H:%M:%S")
        results['metadata']['status'] = 'partial'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Checkpoint salvo: {len(results['queries'])} queries processadas")
    
    def _load_partial_results(self, output_dir, e_radius):
        """Carrega resultados parciais se existirem"""
        filename = f"faiss_benchmark_radius_{e_radius}_partial.json".replace('.', '_')
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"🔄 Continuando de checkpoint: {len(results['queries'])} queries já processadas")
                return results
            except Exception as e:
                print(f"⚠️  Erro ao carregar checkpoint: {e}")
        
        return None
    
    def run_benchmark_for_radius(self, e_radius, queries, load_time, output_dir="faiss_benchmark_results"):
        """
        Executa benchmark para um raio específico com salvamento incremental
        
        Args:
            e_radius (float): Raio para busca FAISS
            queries (dict): Dicionário de QueryFingerprints
            load_time (float): Tempo de carga do banco de referência
            output_dir (str): Diretório para salvar resultados
        
        Returns:
            dict: Resultados do benchmark
        """
        print(f"\n{'='*60}")
        print("E_RADIUS: ", e_radius)
        print(f"EXECUTANDO BENCHMARK PARA e_radius = {e_radius}")
        print(f"{'='*60}")
        
        # Tentar carregar resultados parciais
        partial_results = self._load_partial_results(output_dir, e_radius)
        
        if partial_results:
            results = partial_results
            # Filtrar queries que já foram processadas
            processed_queries = set(results['queries'].keys())
            remaining_queries = {k: v for k, v in queries.items() if k not in processed_queries}
            print(f"🔄 {len(processed_queries)} queries já processadas, {len(remaining_queries)} restantes")
        else:
            # Inicializar nova execução
            results = {
                'e_radius': e_radius,
                'load_time': load_time,
                'queries': {},
                'aggregate_metrics': {},
                'metadata': {
                    'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'total_queries': len(queries),
                    'parameters': {
                        'e_radius': e_radius
                    }
                }
            }
            remaining_queries = queries
        
        # Se não há queries restantes, recalcular métricas e salvar resultado final
        if not remaining_queries:
            print("✅ Todas as queries já foram processadas, recalculando métricas...")
            self._recalculate_aggregate_metrics(results)
            self._save_final_results(results, output_dir, e_radius)
            return results
        
        query_metrics = []
        total_search_time_all_queries = results.get('aggregate_metrics', {}).get('total_search_time_all_queries', 0)
        total_faiss_time = results.get('aggregate_metrics', {}).get('total_faiss_time', 0)
        total_sum_time = results.get('aggregate_metrics', {}).get('total_sum_time', 0)
        
        # Reconstruir query_metrics a partir dos resultados existentes
        for query_name, query_result in results['queries'].items():
            query_metrics.append({
                'query': query_name,
                'precision': query_result['precision'],
                'recall': query_result['recall'],
                'f1_score': query_result['f1_score'],
                'search_time': query_result['total_search_time'],
                'faiss_time': query_result['faiss_time'],
                'sum_time': query_result['sum_time'],
                'true_positives_count': query_result['true_positives_count'],
                'false_positives_count': query_result['false_positives_count'],
                'total_candidates': query_result['total_candidates']
            })
        
        # Processar queries restantes
        checkpoint_interval = 5
        processed_in_this_run = 0
        
        for i, (query_name, query_fp) in enumerate(remaining_queries.items()):
            global_index = len(results['queries']) + 1
            print(f"[{global_index}/{len(queries)}] Processando query: {query_name}")
            
            # Busca no banco de dados
            search_start = time.time()
            try:
                candidates, faiss_time, sum_time = self.db.query_all_windows(
                    query_fp, e_radius=e_radius
                )
                search_time = time.time() - search_start
            except Exception as e:
                print(f"❌ Erro na busca de {query_name}: {e}")
                continue
            
            total_search_time_all_queries += search_time
            total_faiss_time += faiss_time
            total_sum_time += sum_time
            
            # Converter candidatos para conjunto de referências
            returned_references = set(candidates)
            expected_references = self.ground_truth.get(query_name, set())
            
            # Calcular métricas
            true_positives = returned_references & expected_references
            false_positives = returned_references - expected_references
            false_negatives = expected_references - returned_references
            
            # Precision e Recall
            precision = len(true_positives) / len(returned_references) if len(returned_references) > 0 else 0
            recall = len(true_positives) / len(expected_references) if len(expected_references) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Coletar resultados da query
            query_result = {
                'total_search_time': search_time,
                'faiss_time': faiss_time,
                'sum_time': sum_time,
                'total_candidates': len(returned_references),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': list(true_positives),
                'false_positives': list(false_positives),
                'false_negatives': list(false_negatives),
                'true_positives_count': len(true_positives),
                'false_positives_count': len(false_positives),
                'false_negatives_count': len(false_negatives)
            }
            
            results['queries'][query_name] = query_result
            
            query_metrics.append({
                'query': query_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'search_time': search_time,
                'faiss_time': faiss_time,
                'sum_time': sum_time,
                'true_positives_count': len(true_positives),
                'false_positives_count': len(false_positives),
                'total_candidates': len(returned_references)
            })
            
            print(f"  ✅ Recall: {recall:.3f}, Precision: {precision:.3f}, F1: {f1:.3f}")
            print(f"  📊 TP: {len(true_positives)}, FP: {len(false_positives)}, FN: {len(false_negatives)}")
            print(f"  ⏱️  Tempo busca: {search_time:.2f}s, FAISS: {faiss_time:.2f}s, Soma: {sum_time:.2f}s")
            
            processed_in_this_run += 1
            
            # Salvar checkpoint a cada 50 queries ou no final
            if processed_in_this_run % checkpoint_interval == 0 or (i == len(remaining_queries) - 1):
                checkpoint_info = {
                    'processed_queries': len(results['queries']),
                    'total_queries': len(queries),
                    'progress': f"{len(results['queries'])}/{len(queries)}",
                    'checkpoint_time': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Recalcular métricas parciais para o checkpoint
                self._calculate_aggregate_metrics(results, query_metrics, total_search_time_all_queries, 
                                                total_faiss_time, total_sum_time)
                
                self._save_partial_results(results, output_dir, e_radius, checkpoint_info)
        
        # Processamento completo - salvar resultado final
        self._calculate_aggregate_metrics(results, query_metrics, total_search_time_all_queries, 
                                        total_faiss_time, total_sum_time)
        self._save_final_results(results, output_dir, e_radius)
        
        # Remover arquivo parcial se existir
        partial_filename = f"faiss_benchmark_radius_{e_radius}_partial.json".replace('.', '_')
        partial_path = os.path.join(output_dir, partial_filename)
        if os.path.exists(partial_path):
            os.remove(partial_path)
            print(f"🧹 Arquivo parcial removido: {partial_path}")
        
        return results
    
    def _recalculate_aggregate_metrics(self, results):
        """Recalcula métricas agregadas a partir dos resultados existentes"""
        query_metrics = []
        total_search_time_all_queries = 0
        total_faiss_time = 0
        total_sum_time = 0
        
        for query_name, query_result in results['queries'].items():
            query_metrics.append({
                'query': query_name,
                'precision': query_result['precision'],
                'recall': query_result['recall'],
                'f1_score': query_result['f1_score'],
                'search_time': query_result['total_search_time'],
                'faiss_time': query_result['faiss_time'],
                'sum_time': query_result['sum_time'],
                'true_positives_count': query_result['true_positives_count'],
                'false_positives_count': query_result['false_positives_count'],
                'total_candidates': query_result['total_candidates']
            })
            
            total_search_time_all_queries += query_result['total_search_time']
            total_faiss_time += query_result['faiss_time']
            total_sum_time += query_result['sum_time']
        
        self._calculate_aggregate_metrics(results, query_metrics, total_search_time_all_queries, 
                                        total_faiss_time, total_sum_time)
    
    def _save_final_results(self, results, output_dir, e_radius):
        """Salva resultados finais em arquivo JSON"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Nome do arquivo final
        filename = f"faiss_benchmark_radius_{e_radius}.json".replace('.', '_')
        output_path = os.path.join(output_dir, filename)
        
        results['metadata']['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        results['metadata']['status'] = 'completed'
        
        # Remover checkpoint info se existir
        if 'checkpoint' in results:
            del results['checkpoint']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resultados finais salvos em: {output_path}")
        print(f"📊 Total de queries processadas: {len(results['queries'])}")
        
        # Também exportar para CSV
        # self._export_to_csv(results, output_dir, e_radius)
    
    def _calculate_aggregate_metrics(self, results, query_metrics, total_search_time, 
                                   total_faiss_time, total_sum_time):
        """Calcula métricas agregadas"""
        if not query_metrics:
            return
        
        df_metrics = pd.DataFrame(query_metrics)
        
        # Estatísticas básicas
        recall_stats = df_metrics['recall'].describe()
        precision_stats = df_metrics['precision'].describe()
        f1_stats = df_metrics['f1_score'].describe()
        search_time_stats = df_metrics['search_time'].describe()
        faiss_time_stats = df_metrics['faiss_time'].describe()
        sum_time_stats = df_metrics['sum_time'].describe()
        
        # Estatísticas de contagem
        tp_stats = df_metrics['true_positives_count'].describe()
        fp_stats = df_metrics['false_positives_count'].describe()
        total_candidates_stats = df_metrics['total_candidates'].describe()
        
        aggregate = {
            # Métricas de acurácia
            'mean_precision': precision_stats['mean'],
            'mean_recall': recall_stats['mean'],
            'mean_f1_score': f1_stats['mean'],
            
            # Estatísticas detalhadas de recall
            'recall_stats': {
                'mean': recall_stats['mean'],
                'std': recall_stats['std'],
                'min': recall_stats['min'],
                '25%': recall_stats['25%'],
                '50%': recall_stats['50%'],
                '75%': recall_stats['75%'],
                'max': recall_stats['max']
            },
            
            # Estatísticas de contagem
            'tp_stats': {
                'mean': tp_stats['mean'],
                'std': tp_stats['std'],
                'min': tp_stats['min'],
                '25%': tp_stats['25%'],
                '50%': tp_stats['50%'],
                '75%': tp_stats['75%'],
                'max': tp_stats['max']
            },
            
            'fp_stats': {
                'mean': fp_stats['mean'],
                'std': fp_stats['std'],
                'min': fp_stats['min'],
                '25%': fp_stats['25%'],
                '50%': fp_stats['50%'],
                '75%': fp_stats['75%'],
                'max': fp_stats['max']
            },
            
            'total_candidates_stats': {
                'mean': total_candidates_stats['mean'],
                'std': total_candidates_stats['std'],
                'min': total_candidates_stats['min'],
                '25%': total_candidates_stats['25%'],
                '50%': total_candidates_stats['50%'],
                '75%': total_candidates_stats['75%'],
                'max': total_candidates_stats['max']
            },
            
            # Estatísticas de tempo
            'search_time_stats': {
                'mean': search_time_stats['mean'],
                'std': search_time_stats['std'],
                'min': search_time_stats['min'],
                '25%': search_time_stats['25%'],
                '50%': search_time_stats['50%'],
                '75%': search_time_stats['75%'],
                'max': search_time_stats['max']
            },
            
            'faiss_time_stats': {
                'mean': faiss_time_stats['mean'],
                'std': faiss_time_stats['std'],
                'min': faiss_time_stats['min'],
                '25%': faiss_time_stats['25%'],
                '50%': faiss_time_stats['50%'],
                '75%': faiss_time_stats['75%'],
                'max': faiss_time_stats['max']
            },
            
            'sum_time_stats': {
                'mean': sum_time_stats['mean'],
                'std': sum_time_stats['std'],
                'min': sum_time_stats['min'],
                '25%': sum_time_stats['25%'],
                '50%': sum_time_stats['50%'],
                '75%': sum_time_stats['75%'],
                'max': sum_time_stats['max']
            },
            
            # Totais
            'total_queries': len(query_metrics),
            'total_search_time_all_queries': total_search_time,
            'total_faiss_time': total_faiss_time,
            'total_sum_time': total_sum_time,
            'successful_queries': len([q for q in query_metrics if q['recall'] > 0])
        }
        
        results['aggregate_metrics'] = aggregate
    
    def _export_to_csv(self, results, output_dir, e_radius):
        """Exporta resultados agregados para CSV"""
        agg = results.get('aggregate_metrics', {})
        recall_stats = agg.get('recall_stats', {})
        tp_stats = agg.get('tp_stats', {})
        fp_stats = agg.get('fp_stats', {})
        
        # Criar DataFrame no formato solicitado
        data = {
            'Metric': ['RECALL', '', '', '', '', '', '', 
                      'TP(qtd)', '', '', '', '', '',
                      'FP(qtd)', '', '', '', '', '',
                      'TOTAL_CANDIDATES', '', '', '', '', ''],
            'Statistic': ['mean', 'std', 'min', '25', '50', '75', 'max',
                         'mean', 'std', 'min', '25', '50', '75', 'max',
                         'mean', 'std', 'min', '25', '50', '75', 'max',
                         'mean', 'std', 'min', '25', '50', '75', 'max'],
            'HNSW': [
                # Recall
                recall_stats.get('mean', 0), recall_stats.get('std', 0), 
                recall_stats.get('min', 0), recall_stats.get('25%', 0), 
                recall_stats.get('50%', 0), recall_stats.get('75%', 0), 
                recall_stats.get('max', 0),
                # True Positives
                tp_stats.get('mean', 0), tp_stats.get('std', 0),
                tp_stats.get('min', 0), tp_stats.get('25%', 0),
                tp_stats.get('50%', 0), tp_stats.get('75%', 0),
                tp_stats.get('max', 0),
                # False Positives
                fp_stats.get('mean', 0), fp_stats.get('std', 0),
                fp_stats.get('min', 0), fp_stats.get('25%', 0),
                fp_stats.get('50%', 0), fp_stats.get('75%', 0),
                fp_stats.get('max', 0),
                # Total Candidates
                agg.get('total_candidates_stats', {}).get('mean', 0),
                agg.get('total_candidates_stats', {}).get('std', 0),
                agg.get('total_candidates_stats', {}).get('min', 0),
                agg.get('total_candidates_stats', {}).get('25%', 0),
                agg.get('total_candidates_stats', {}).get('50%', 0),
                agg.get('total_candidates_stats', {}).get('75%', 0),
                agg.get('total_candidates_stats', {}).get('max', 0)
            ]
        }

        df = pd.DataFrame(data)
        csv_filename = f"faiss_benchmark_radius_{e_radius:.2f}.csv".replace('.', '_')
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"📈 CSV exportado para: {csv_path}")
    
    def generate_summary_report(self, all_results, output_dir):
        """Gera um relatório resumido comparando todos os raios testados"""
        summary_data = []
        
        for e_radius, results in all_results.items():
            agg = results.get('aggregate_metrics', {})
            recall_stats = agg.get('recall_stats', {})
            fp_stats = agg.get('fp_stats', {})
            
            summary_data.append({
                'e_radius': e_radius,
                'mean_recall': recall_stats.get('mean', 0),
                'std_recall': recall_stats.get('std', 0),
                'mean_fp': fp_stats.get('mean', 0),
                'std_fp': fp_stats.get('std', 0),
                'total_queries': agg.get('total_queries', 0),
                'total_search_time': agg.get('total_search_time_all_queries', 0),
                'total_faiss_time': agg.get('total_faiss_time', 0),
                'total_sum_time': agg.get('total_sum_time', 0),
                'load_time': results.get('load_time', 0)
            })
        
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "faiss_benchmark_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        
        print(f"\n📊 Relatório sumário salvo em: {summary_path}")
        print("\n" + "="*80)
        print("RESUMO DO BENCHMARK - COMPARAÇÃO ENTRE RAIO")
        print("="*80)
        print(df_summary.round(4).to_string(index=False))
        
        return df_summary

def main():
    """Função principal para executar o benchmark incremental de raios"""
    
    # Configurações
    GROUND_TRUTH_CSV = "/mnt/disk1/BAF/metadata/cross_annotations.csv"
    REFERENCES_DIR = "/mnt/disk1/BAF/qfp_features/references"
    QUERIES_PICKLE_DIR = "/mnt/disk1/BAF/qfp_features/queries"  # Diretório com pickles das queries
    OUTPUT_DIR = "/home/luiz/repositories/qfp/qfp/benchmark/data/hnsw_radius"
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Benchmark simplificado do FAISS com múltiplos raios')
    parser.add_argument('--max_queries', type=int, default=15,
                       help='Número máximo de queries a serem processadas (padrão: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed para reproducibilidade')
    parser.add_argument('--start_radius', type=float, default=0.0000001,
                       help='Raio inicial (padrão: 0.02)')
    parser.add_argument('--end_radius', type=float, default=1.0,
                       help='Raio final (padrão: 1.0)')
    parser.add_argument('--step_radius', type=float, default=0.00000001,
                       help='Incremento do raio (padrão: 0.01)')
    
    args = parser.parse_args()
    
    # Inicializar benchmark
    benchmark = FAISSBenchmark(
        ground_truth_csv=GROUND_TRUTH_CSV,
        references_dir=REFERENCES_DIR,
        queries_pickle_dir=QUERIES_PICKLE_DIR
    )
    
    # Carregar queries
    print("Carregando fingerprints de query...")
    queries = benchmark.load_query_fingerprints(
        max_queries=args.max_queries,
        random_seed=args.seed
    )
    
    if not queries:
        print("❌ Nenhuma query válida encontrada!")
        return
    
    # CONSTRUIR BANCO UMA ÚNICA VEZ ANTES DO LOOP
    print("Construindo banco de dados de referência (uma única vez)...")
    load_time = benchmark.build_reference_database()
    
    # Executar benchmark para cada raio
    all_results = {}
    radii = np.arange(args.start_radius, args.end_radius + args.step_radius, args.step_radius)
    print("RADII: ", radii)
    
    for e_radius in radii:
        # Passar load_time como parâmetro
        results = benchmark.run_benchmark_for_radius(e_radius, queries, load_time, OUTPUT_DIR)
        all_results[e_radius] = results
    
    # Gerar relatório sumário
    benchmark.generate_summary_report(all_results, OUTPUT_DIR)
    
    print(f"\n🎯 Benchmark concluído! Resultados salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()