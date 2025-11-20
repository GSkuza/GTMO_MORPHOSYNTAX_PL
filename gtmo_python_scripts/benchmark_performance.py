#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GTMØ Performance Benchmarks
============================
Comprehensive performance benchmarks for GTMØ Quantum Morphosyntax Engine
with property-based testing validation.
"""

import sys
import os
import time
import json
import statistics
import memory_profiler
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    import numpy as np
    from gtmo_morphosyntax import QuantumMorphosyntaxEngine
    from test_property_based import PropertyBasedTestSuite, create_golden_dataset
    GTMO_AVAILABLE = True
except ImportError as e:
    print(f"❌ GTMØ modules not available: {e}")
    GTMO_AVAILABLE = False

class GTMOBenchmarkSuite:
    """Comprehensive benchmark suite for GTMØ performance testing."""
    
    def __init__(self):
        self.engine = QuantumMorphosyntaxEngine() if GTMO_AVAILABLE else None
        self.results = {}
        self.memory_usage = []
        
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            return result, execution_time, None
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            return None, execution_time, str(e)
    
    @memory_profiler.profile
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        return func(*args, **kwargs)
    
    def benchmark_basic_analysis(self):
        """Benchmark basic morphosyntactic analysis."""
        print("\n🔬 Benchmarking basic analysis...")
        
        test_texts = [
            "Kot śpi.",  # 2 words
            "Rzeczpospolita Polska jest demokratycznym państwem prawnym.",  # 7 words
            "Nieantykonstytucjonalizacyjność polega na braku zgodności z konstytucją Rzeczypospolitej Polskiej.",  # 11 words
            "Mimo że wczoraj, gdy szliśmy przez park, w którym rosną stare dęby, spotkaliśmy grupę studentów prowadzących ożywioną dyskusję." * 2  # Long text
        ]
        
        results = []
        
        for i, text in enumerate(test_texts):
            word_count = len(text.split())
            
            # Measure execution time
            result, exec_time, error = self.measure_execution_time(
                self.engine.gtmo_analyze_quantum, text
            )
            
            if not error:
                # Measure memory usage
                try:
                    mem_usage = memory_profiler.memory_usage(
                        (self.engine.gtmo_analyze_quantum, (text,))
                    )
                    max_memory = max(mem_usage) if mem_usage else 0
                    memory_delta = max(mem_usage) - min(mem_usage) if len(mem_usage) > 1 else 0
                except:
                    max_memory = 0
                    memory_delta = 0
                
                benchmark_result = {
                    'test_id': f'basic_{i}',
                    'word_count': word_count,
                    'text_length': len(text),
                    'execution_time_ms': exec_time * 1000,
                    'max_memory_mb': max_memory,
                    'memory_delta_mb': memory_delta,
                    'throughput_words_per_sec': word_count / exec_time if exec_time > 0 else 0,
                    'error': None
                }
                
                print(f"  📊 {word_count:3d} words: {exec_time*1000:6.1f}ms, {word_count/exec_time:5.1f} words/sec")
                
            else:
                benchmark_result = {
                    'test_id': f'basic_{i}',
                    'word_count': word_count,
                    'text_length': len(text),
                    'execution_time_ms': exec_time * 1000,
                    'error': error
                }
                print(f"  ❌ {word_count:3d} words: FAILED ({error})")
            
            results.append(benchmark_result)
        
        self.results['basic_analysis'] = results
        return results
    
    def benchmark_constitutional_metrics(self):
        """Benchmark constitutional metrics calculation."""
        print("\n⚖️  Benchmarking constitutional metrics...")
        
        test_cases = [
            ("Simple", "Kot ma dom."),
            ("Medium", "Rzeczpospolita Polska przestrzega prawa międzynarodowego."),
            ("Complex", "Najwybitniejszymi naukowcami zachwycali się studenci uniwersytetu."),
            ("Constitutional", "Sąd Konstytucyjny orzeka o zgodności ustaw z Konstytucją Rzeczypospolitej Polskiej w zakresie określonym w niniejszej ustawie."),
        ]
        
        results = []
        
        for case_name, text in test_cases:
            result, exec_time, error = self.measure_execution_time(
                self.engine.gtmo_analyze_quantum, text
            )
            
            if not error:
                # Check duality accuracy
                const_metrics = result['constitutional_metrics']
                ci = const_metrics['indefiniteness']['value']
                cd = const_metrics['definiteness']['value']
                duality_error = const_metrics['duality']['error_percent']
                
                benchmark_result = {
                    'case': case_name,
                    'execution_time_ms': exec_time * 1000,
                    'duality_error_percent': duality_error,
                    'ci_value': ci,
                    'cd_value': cd,
                    'semantic_accessibility': const_metrics['semantic_accessibility']['value'],
                    'duality_verification': 'PASSED' if duality_error < 1.0 else 'WARNING'
                }
                
                print(f"  📈 {case_name:12s}: {exec_time*1000:6.1f}ms, duality_error={duality_error:.3f}%")
                
            else:
                benchmark_result = {
                    'case': case_name,
                    'execution_time_ms': exec_time * 1000,
                    'error': error
                }
                print(f"  ❌ {case_name:12s}: FAILED ({error})")
            
            results.append(benchmark_result)
        
        self.results['constitutional_metrics'] = results
        return results
    
    def benchmark_quantum_analysis(self):
        """Benchmark quantum superposition analysis."""
        print("\n🌀 Benchmarking quantum analysis...")
        
        # Test quantum superposition with ambiguous texts
        quantum_texts = [
            "Słowo może znaczyć różne rzeczy.",
            "Wieloznaczność tworzy superpozycję semantyczną stanów kwantowych.",
            "Bank rzeki oraz bank finansowy demonstrują kwantową naturę języka.",
            "Zamek na drzwiach i zamek królewski pokazują entanglement znaczeń."
        ]
        
        results = []
        
        for i, text in enumerate(quantum_texts):
            result, exec_time, error = self.measure_execution_time(
                self.engine.gtmo_analyze_quantum, text
            )
            
            if not error:
                quantum_metrics = result['quantum_metrics']
                
                benchmark_result = {
                    'test_id': f'quantum_{i}',
                    'execution_time_ms': exec_time * 1000,
                    'quantum_coherence': quantum_metrics['total_coherence'],
                    'quantum_words': quantum_metrics['quantum_words'],
                    'entanglements': quantum_metrics['entanglements'],
                    'superposition_type': quantum_metrics['superposition_type'],
                    'ambiguity': result['additional_metrics']['ambiguity']
                }
                
                print(f"  🌀 Test {i}: {exec_time*1000:6.1f}ms, coherence={quantum_metrics['total_coherence']:.3f}")
                
            else:
                benchmark_result = {
                    'test_id': f'quantum_{i}',
                    'execution_time_ms': exec_time * 1000,
                    'error': error
                }
                print(f"  ❌ Test {i}: FAILED ({error})")
            
            results.append(benchmark_result)
        
        self.results['quantum_analysis'] = results
        return results
    
    def benchmark_scaling_performance(self):
        """Benchmark performance scaling with text length."""
        print("\n📈 Benchmarking scaling performance...")
        
        base_text = "Rzeczpospolita Polska jest demokratycznym państwem prawnym urzeczywistniającym zasady sprawiedliwości społecznej."
        
        # Test different text lengths
        scale_factors = [1, 2, 5, 10, 20]
        results = []
        
        for scale in scale_factors:
            scaled_text = (base_text + " ") * scale
            word_count = len(scaled_text.split())
            
            # Multiple runs for statistical significance
            execution_times = []
            for run in range(3):
                _, exec_time, error = self.measure_execution_time(
                    self.engine.gtmo_analyze_quantum, scaled_text
                )
                
                if not error:
                    execution_times.append(exec_time)
                else:
                    print(f"  ❌ Scale {scale}, run {run}: FAILED")
                    break
            
            if execution_times:
                avg_time = statistics.mean(execution_times)
                std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                
                benchmark_result = {
                    'scale_factor': scale,
                    'word_count': word_count,
                    'avg_execution_time_ms': avg_time * 1000,
                    'std_execution_time_ms': std_time * 1000,
                    'throughput_words_per_sec': word_count / avg_time,
                    'runs': len(execution_times)
                }
                
                print(f"  📊 Scale {scale:2d} ({word_count:3d} words): {avg_time*1000:6.1f}±{std_time*1000:4.1f}ms")
                
            else:
                benchmark_result = {
                    'scale_factor': scale,
                    'word_count': word_count,
                    'error': 'All runs failed'
                }
            
            results.append(benchmark_result)
        
        self.results['scaling_performance'] = results
        return results
    
    def benchmark_property_based_tests(self):
        """Benchmark property-based test execution."""
        print("\n🧪 Benchmarking property-based tests...")
        
        if not hasattr(sys.modules.get('test_property_based'), 'PropertyBasedTestSuite'):
            print("  ⚠️  Property-based test suite not available")
            return []
        
        test_suite = PropertyBasedTestSuite()
        
        # Measure time for property-based test execution
        result, exec_time, error = self.measure_execution_time(
            test_suite.run_all_tests
        )
        
        benchmark_result = {
            'total_execution_time_ms': exec_time * 1000,
            'tests_passed': result if not error else False,
            'failed_tests': len(test_suite.failed_tests) if hasattr(test_suite, 'failed_tests') else 0,
            'error': error
        }
        
        if not error:
            print(f"  🧪 Property tests: {exec_time*1000:6.1f}ms, passed={result}")
        else:
            print(f"  ❌ Property tests: FAILED ({error})")
        
        self.results['property_based_tests'] = benchmark_result
        return benchmark_result
    
    def analyze_performance_trends(self):
        """Analyze performance trends and generate insights."""
        print("\n📊 Analyzing performance trends...")
        
        insights = []
        
        # Analyze basic analysis performance
        if 'basic_analysis' in self.results:
            basic_results = [r for r in self.results['basic_analysis'] if 'error' not in r or r['error'] is None]
            if basic_results:
                avg_throughput = statistics.mean([r['throughput_words_per_sec'] for r in basic_results])
                insights.append(f"Average throughput: {avg_throughput:.1f} words/second")
                
                # Find performance sweet spot
                best_result = max(basic_results, key=lambda x: x['throughput_words_per_sec'])
                insights.append(f"Best performance: {best_result['throughput_words_per_sec']:.1f} words/sec ({best_result['word_count']} words)")
        
        # Analyze constitutional metrics accuracy
        if 'constitutional_metrics' in self.results:
            const_results = [r for r in self.results['constitutional_metrics'] if 'error' not in r or r['error'] is None]
            if const_results:
                avg_duality_error = statistics.mean([r['duality_error_percent'] for r in const_results])
                insights.append(f"Average duality error: {avg_duality_error:.4f}%")
                
                passed_duality = sum(1 for r in const_results if r['duality_verification'] == 'PASSED')
                insights.append(f"Duality verification rate: {passed_duality}/{len(const_results)} ({passed_duality/len(const_results)*100:.1f}%)")
        
        # Analyze scaling behavior
        if 'scaling_performance' in self.results:
            scale_results = [r for r in self.results['scaling_performance'] if 'error' not in r or r['error'] is None]
            if len(scale_results) >= 2:
                # Check if performance scales linearly
                throughputs = [r['throughput_words_per_sec'] for r in scale_results]
                if len(throughputs) > 1:
                    throughput_trend = "declining" if throughputs[-1] < throughputs[0] else "stable"
                    insights.append(f"Scaling behavior: {throughput_trend} throughput with text length")
        
        # Performance classification
        if 'basic_analysis' in self.results:
            basic_results = [r for r in self.results['basic_analysis'] if 'error' not in r or r['error'] is None]
            if basic_results:
                avg_time_per_word = statistics.mean([r['execution_time_ms'] / r['word_count'] for r in basic_results])
                
                if avg_time_per_word < 10:
                    performance_class = "🚀 EXCELLENT"
                elif avg_time_per_word < 50:
                    performance_class = "✅ GOOD"  
                elif avg_time_per_word < 100:
                    performance_class = "⚠️  ACCEPTABLE"
                else:
                    performance_class = "❌ NEEDS OPTIMIZATION"
                
                insights.append(f"Performance classification: {performance_class}")
                insights.append(f"Average time per word: {avg_time_per_word:.1f}ms")
        
        for insight in insights:
            print(f"  💡 {insight}")
        
        self.results['insights'] = insights
        return insights
    
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report."""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        report = {
            'metadata': {
                'timestamp': timestamp,
                'engine_version': '2.0',
                'benchmark_version': '1.0',
                'system_info': {
                    'platform': sys.platform,
                    'python_version': sys.version
                }
            },
            'benchmark_results': self.results,
            'summary': {
                'total_benchmarks': len(self.results),
                'execution_timestamp': timestamp
            }
        }
        
        # Save report
        report_file = f"benchmark_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Benchmark report saved: {report_file}")
        return report
    
    def run_comprehensive_benchmarks(self):
        """Run all benchmarks and generate report."""
        print("🚀 GTMØ COMPREHENSIVE PERFORMANCE BENCHMARKS")
        print("=" * 80)
        
        if not GTMO_AVAILABLE:
            print("❌ GTMØ engine not available")
            return False
        
        try:
            # Run all benchmark suites
            self.benchmark_basic_analysis()
            self.benchmark_constitutional_metrics()
            self.benchmark_quantum_analysis()
            self.benchmark_scaling_performance()
            self.benchmark_property_based_tests()
            
            # Analyze trends
            self.analyze_performance_trends()
            
            # Generate report
            report = self.generate_benchmark_report()
            
            print("\n" + "=" * 80)
            print("✅ BENCHMARK SUITE COMPLETED")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"❌ Benchmark suite failed: {e}")
            traceback.print_exc()
            return False


def main():
    """Main benchmark execution."""
    benchmark_suite = GTMOBenchmarkSuite()
    return benchmark_suite.run_comprehensive_benchmarks()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)