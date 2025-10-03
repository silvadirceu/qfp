# benchmark_incremental.py
import json
import pandas as pd
import numpy as np
import time
import os
import glob
from collections import defaultdict
from qfp.db_mem import InMemoryQfpDB
from qfp import QueryFingerprint

class QFPBenchmarkIncremental:
    def __init__(self, ground_truth_csv, references_dir, queries_dir):
        """
        Inicializa o benchmark com salvamento incremental
        
        Args:
            ground_truth_csv (str): Caminho para o arquivo CSV com anotações
            references_dir (str): Diretório com fingerprints de referência (.pkl)
            queries_dir (str): Diretório com arquivos de áudio de query (.wav)
        """
        self.ground_truth_csv = ground_truth_csv
        self.references_dir = references_dir
        self.queries_dir = queries_dir
        self.db = InMemoryQfpDB()
        self.ground_truth = None
        self.results = {}
        self.processed_queries = set()
        
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
        """Constrói o banco de dados de referência"""
        print("Construindo banco de dados de referência...")
        start_time = time.time()
        
        self.db.store_all_pickles_from_directory(self.references_dir)
        
        build_time = time.time() - start_time
        print(f"Banco construído em {build_time:.2f} segundos")
        print(f"Total de referências: {len(self.db.fingerprints)}")
        
        return build_time
    
    def load_existing_results(self, results_file):
        """Carrega resultados existentes para continuar de onde parou"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            self.results = existing_results
            self.processed_queries = set(existing_results.get('queries', {}).keys())
            
            print(f"Carregados {len(self.processed_queries)} queries processadas anteriormente")
            return True
        except FileNotFoundError:
            print("Nenhum resultado anterior encontrado. Iniciando do zero.")
            return False
        except Exception as e:
            print(f"Erro ao carregar resultados existentes: {e}")
            return False
    
    def run_benchmark(self, vThreshold=0.05, e_radius=0.2, queries=None, 
                     output_file="benchmark_results.json", save_interval=50):
        """
        Executa o benchmark completo com salvamento incremental
        
        Args:
            vThreshold (float): Threshold de validação
            e_radius (float): Raio para busca FAISS
            queries (list): Lista específica de queries para testar (opcional)
            output_file (str): Arquivo para salvar resultados
            save_interval (int): Salvar a cada N queries
        """
        if queries is None:
            # Usa todas as queries do ground truth
            queries = list(self.ground_truth.keys())
        
        # Remove queries já processadas
        remaining_queries = [q for q in queries if q not in self.processed_queries]
        
        if not remaining_queries:
            print("Todas as queries já foram processadas!")
            return self._finalize_results()
        
        print(f"Processando {len(remaining_queries)} queries restantes...")
        
        # Inicializar estrutura de resultados se não existir
        if not self.results:
            self.results = {
                'build_time': 0,
                'queries': {},
                'aggregate_metrics': {},
                'metadata': {
                    'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'total_queries': len(queries),
                    'processed_queries': 0,
                    'parameters': {
                        'vThreshold': vThreshold,
                        'e_radius': e_radius
                    }
                }
            }
        
        # Construir banco de dados se não foi feito ainda
        if self.results['build_time'] == 0:
            build_time = self.build_reference_database()
            self.results['build_time'] = build_time
        else:
            print("Banco de dados já construído anteriormente")
        
        query_metrics = []
        batch_count = 0
        
        for i, query_name in enumerate(remaining_queries):
            print(f"\n[{i+1}/{len(remaining_queries)}] Processando query: {query_name}")
            
            query_path = os.path.join(self.queries_dir, f"{query_name}.wav")
            if not os.path.exists(query_path):
                print(f"Arquivo de query não encontrado: {query_path}")
                continue
            
            # Extração de features da query
            extraction_start = time.time()
            try:
                query_fp = QueryFingerprint(query_path)
                query_fp.create()
                extraction_time = time.time() - extraction_start
            except Exception as e:
                print(f"Erro na extração de {query_name}: {e}")
                self._save_failed_query(query_name, str(e))
                continue
            
            # Busca no banco de dados
            search_start = time.time()
            try:
                matches, faiss_time, filter_time, hist_time, match_time = self.db.query(
                    query_fp, vThreshold=vThreshold, e_radius=e_radius
                )
                search_time = time.time() - search_start
            except Exception as e:
                print(f"Erro na busca de {query_name}: {e}")
                self._save_failed_query(query_name, str(e))
                continue
            
            # Ordenar matches por score
            sorted_matches = sorted(matches, key=lambda m: m.vScore, reverse=True)
            
            # Calcular métricas de precisão/recall
            precision, recall, f1, true_positives, false_positives, false_negatives = self._calculate_metrics(
                query_name, sorted_matches
            )
            
            # Coletar resultados
            query_result = {
                'extraction_time': extraction_time,
                'total_search_time': search_time,
                'faiss_search_time': faiss_time,
                'filter_time': filter_time,
                'process_histogram_time': hist_time,
                'matches_time': match_time,
                'total_matches': len(sorted_matches),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'matches': [{
                    'record': m.record,
                    'offset': m.offset,
                    'vScore': m.vScore
                } for m in sorted_matches[:10]]  # Salva apenas top 10 matches
            }
            
            # Adicionar aos resultados
            self.results['queries'][query_name] = query_result
            self.processed_queries.add(query_name)
            
            query_metrics.append({
                'query': query_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'extraction_time': extraction_time,
                'search_time': search_time
            })
            
            print(f"  ✅ Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            print(f"  ⏱️  Tempo extração: {extraction_time:.2f}s, Tempo busca: {search_time:.2f}s")
            
            # Salvamento incremental
            batch_count += 1
            if batch_count >= save_interval:
                self._save_incremental_results(output_file)
                batch_count = 0
                print(f"💾 Resultados salvos incrementalmente ({len(self.processed_queries)} queries processadas)")
        
        # Salvar resultados finais
        self._save_incremental_results(output_file)
        
        # Calcular métricas agregadas
        self._calculate_aggregate_metrics(query_metrics)
        
        return self.results
    
    def _save_failed_query(self, query_name, error_message):
        """Registra uma query que falhou"""
        self.results['queries'][query_name] = {
            'error': error_message,
            'status': 'failed'
        }
        self.processed_queries.add(query_name)
    
    def _save_incremental_results(self, output_file):
        """Salva resultados de forma incremental"""
        try:
            # Atualizar metadados
            self.results['metadata']['last_update'] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.results['metadata']['processed_queries'] = len(self.processed_queries)
            self.results['metadata']['progress'] = f"{len(self.processed_queries)}/{self.results['metadata']['total_queries']}"
            
            # Salvar em arquivo temporário primeiro
            temp_file = output_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            # Substituir arquivo original
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rename(temp_file, output_file)
            
        except Exception as e:
            print(f"❌ Erro ao salvar resultados incrementais: {e}")
    
    def _calculate_metrics(self, query_name, matches):
        """Calcula precision, recall e F1-score para uma query"""
        expected_matches = self.ground_truth.get(query_name, set())
        returned_matches = {match.record for match in matches}
        
        true_positives = returned_matches & expected_matches
        false_positives = returned_matches - expected_matches
        false_negatives = expected_matches - returned_matches
        
        # Precision: TP / (TP + FP)
        precision = len(true_positives) / len(returned_matches) if len(returned_matches) > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall = len(true_positives) / len(expected_matches) if len(expected_matches) > 0 else 0
        
        # F1-score: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1, list(true_positives), list(false_positives), list(false_negatives)
    
    def _calculate_aggregate_metrics(self, query_metrics):
        """Calcula métricas agregadas sobre todas as queries"""
        if not query_metrics:
            return
        
        df_metrics = pd.DataFrame(query_metrics)
        
        aggregate = {
            'mean_precision': df_metrics['precision'].mean(),
            'mean_recall': df_metrics['recall'].mean(),
            'mean_f1_score': df_metrics['f1_score'].mean(),
            'std_precision': df_metrics['precision'].std(),
            'std_recall': df_metrics['recall'].std(),
            'std_f1_score': df_metrics['f1_score'].std(),
            'mean_extraction_time': df_metrics['extraction_time'].mean(),
            'mean_search_time': df_metrics['search_time'].mean(),
            'total_queries': len(self.processed_queries),
            'successful_queries': len([q for q in self.results['queries'].values() 
                                     if 'f1_score' in q and q['f1_score'] > 0])
        }
        
        self.results['aggregate_metrics'] = aggregate
    
    def _finalize_results(self):
        """Finaliza os resultados quando todas as queries foram processadas"""
        self.results['metadata']['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.results['metadata']['status'] = 'completed'
        return self.results
    
    def get_progress(self):
        """Retorna o progresso atual"""
        total = self.results.get('metadata', {}).get('total_queries', 0)
        processed = len(self.processed_queries)
        progress_pct = (processed / total * 100) if total > 0 else 0
        
        return {
            'processed': processed,
            'total': total,
            'progress_percentage': progress_pct,
            'remaining': total - processed
        }
    
    def generate_report(self):
        """Gera relatório resumido do benchmark"""
        if not self.results:
            print("Nenhum resultado disponível. Execute o benchmark primeiro.")
            return
        
        agg = self.results.get('aggregate_metrics', {})
        metadata = self.results.get('metadata', {})
        
        print("\n" + "="*60)
        print("RELATÓRIO DE BENCHMARK - QFP SYSTEM")
        print("="*60)
        print(f"Progresso: {metadata.get('progress', 'N/A')}")
        print(f"Status: {metadata.get('status', 'running')}")
        print(f"Tempo construção banco: {self.results.get('build_time', 0):.2f}s")
        
        if agg:
            print(f"Total de queries processadas: {agg.get('total_queries', 0)}")
            print(f"Queries com match: {agg.get('successful_queries', 0)}")
            success_rate = (agg.get('successful_queries', 0) / agg.get('total_queries', 1) * 100)
            print(f"Taxa de sucesso: {success_rate:.1f}%")
            print("\n--- MÉTRICAS DE ACURÁCIA ---")
            print(f"Precision média: {agg.get('mean_precision', 0):.3f} (±{agg.get('std_precision', 0):.3f})")
            print(f"Recall médio:    {agg.get('mean_recall', 0):.3f} (±{agg.get('std_recall', 0):.3f})")
            print(f"F1-Score médio:  {agg.get('mean_f1_score', 0):.3f} (±{agg.get('std_f1_score', 0):.3f})")
            print("\n--- MÉTRICAS DE TEMPO ---")
            print(f"Tempo extração médio: {agg.get('mean_extraction_time', 0):.2f}s")
            print(f"Tempo busca médio:    {agg.get('mean_search_time', 0):.2f}s")
        else:
            print("Métricas agregadas ainda não disponíveis")
        
        print("="*60)

# Função principal para executar o benchmark com continuidade
def main():
    # Configurações
    GROUND_TRUTH_CSV = "/mnt/disk1/BAF/metadata/cross_annotations.csv"
    REFERENCES_DIR = "/mnt/disk1/BAF/qfp_features/references"
    QUERIES_DIR = "/mnt/disk1/BAF/audio/queries"
    OUTPUT_FILE = "qfp_benchmark_results.json"
    SAVE_INTERVAL = 50  # Salvar a cada 50 queries
    
    # Inicializar benchmark
    benchmark = QFPBenchmarkIncremental(
        ground_truth_csv=GROUND_TRUTH_CSV,
        references_dir=REFERENCES_DIR,
        queries_dir=QUERIES_DIR
    )
    
    # Tentar carregar resultados existentes
    resume = benchmark.load_existing_results(OUTPUT_FILE)
    
    if resume:
        progress = benchmark.get_progress()
        print(f"Progresso atual: {progress['processed']}/{progress['total']} queries ({progress['progress_percentage']:.1f}%)")
        
        continuar = input("Deseja continuar de onde parou? (s/n): ").strip().lower()
        if continuar != 's':
            print("Reiniciando benchmark...")
            benchmark = QFPBenchmarkIncremental(
                ground_truth_csv=GROUND_TRUTH_CSV,
                references_dir=REFERENCES_DIR,
                queries_dir=QUERIES_DIR
            )
    
    # Executar benchmark
    print("Iniciando benchmark do sistema QFP...")
    results = benchmark.run_benchmark(
        vThreshold=0.05,
        e_radius=0.2,
        output_file=OUTPUT_FILE,
        save_interval=SAVE_INTERVAL
    )
    
    # Gerar relatório final
    benchmark.generate_report()
    
    print(f"\n✅ Benchmark concluído! Resultados salvos em: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()