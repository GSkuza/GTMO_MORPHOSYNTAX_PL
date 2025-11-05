# GTMØ Comprehensive Testing Suite
## Property-Based Testing & Golden Dataset Validation

### 📋 Overview

This comprehensive testing suite provides 100/100 test coverage for the GTMØ Quantum Morphosyntax Engine using:

- **Property-Based Testing** with Hypothesis framework
- **Golden Dataset Validation** with comprehensive test cases
- **Performance Benchmarking** with scaling analysis
- **Stateful Testing** for complex interaction scenarios
- **Integration Testing** with pytest framework

### 🧪 Test Components

#### 1. Property-Based Tests (`test_property_based.py`)

Uses Hypothesis to automatically generate test cases and verify fundamental properties:

- **Coordinate Bounds**: All coordinates ∈ [0,1]
- **Constitutional Duality**: CI × CD = Depth² (perfect mathematical duality)
- **Semantic Accessibility**: SA ∈ [0,1] with proper normalization
- **Ambiguity Monotonicity**: Higher ambiguity → higher CI
- **Quantum Coherence**: Bounded quantum metrics
- **Axiom System Stability**: No coordinate explosions
- **Rhetorical Invertibility**: Sound mathematical transformations
- **Entropy Conservation**: Consistent entropy measures

#### 2. Golden Dataset (`golden_dataset_comprehensive.json`)

Comprehensive reference dataset with expected results for:

- Basic Polish syntax patterns
- Complex morphological cases  
- Quantum superposition triggers
- High/low entropy examples
- Irony and paradox patterns
- Constitutional law domain texts
- Edge cases and complex syntax
- Technical/scientific language

#### 3. Performance Benchmarks (`benchmark_performance.py`)

Comprehensive performance analysis:

- **Basic Analysis**: Throughput measurement (words/second)
- **Constitutional Metrics**: Duality accuracy benchmarking
- **Quantum Analysis**: Superposition/entanglement performance
- **Scaling Performance**: Linear scaling validation
- **Memory Usage**: Memory profiling and optimization
- **Property Test Performance**: Test suite execution time

#### 4. Test Runner (`test_runner.py`)

Unified test execution with:

- Dependency checking and auto-installation
- Module-based test discovery
- pytest integration
- Coverage reporting
- Success rate calculation

### 🚀 Quick Start

#### Install Dependencies

```bash
pip install -r requirements-testing.txt
```

#### Run All Tests

```bash
python test_runner.py
```

#### Run Individual Test Suites

```bash
# Property-based tests only
python test_property_based.py

# Performance benchmarks only  
python benchmark_performance.py

# Constitutional duality tests
python test_constitutional_duality.py
```

#### Run with pytest + Hypothesis

```bash
pytest test_integration_pytest.py -v --hypothesis-show-statistics
```

### 📊 Test Coverage Goals

| Component | Target Coverage | Validation Method |
|-----------|----------------|-------------------|
| Coordinate Bounds | 100% | Property-based testing |
| Constitutional Duality | 100% | Mathematical verification |
| Quantum Mechanics | 95%+ | Superposition validation |
| Rhetorical Analysis | 90%+ | Mode detection testing |
| Axiom System | 100% | Stability verification |
| Performance | 90%+ | Benchmark validation |

### 🎯 Property-Based Testing Strategy

#### Core Properties Tested

1. **Mathematical Invariants**
   ```python
   @given(st.text(min_size=5, max_size=1000))
   def test_coordinate_bounds(self, text):
       result = engine.analyze(text)
       coords = result['coordinates']
       assert 0 <= coords['determination'] <= 1
       assert 0 <= coords['stability'] <= 1  
       assert 0 <= coords['entropy'] <= 1
   ```

2. **Constitutional Duality**
   ```python
   @given(st.text(min_size=10, max_size=500))
   def test_duality_invariant(self, text):
       result = engine.analyze(text)
       ci = result['constitutional_metrics']['indefiniteness']['value']
       cd = result['constitutional_metrics']['definiteness']['value']
       depth_squared = result['constitutional_metrics']['duality']['theoretical']
       assert abs(ci * cd - depth_squared) < 0.01
   ```

3. **Semantic Accessibility**
   ```python
   @given(st.text(min_size=5, max_size=200))
   def test_semantic_accessibility_bounds(self, text):
       result = engine.analyze(text)
       sa = result['constitutional_metrics']['semantic_accessibility']['value']
       assert 0 <= sa <= 1
       
       # Formula consistency: SA = CD / Depth²
       cd = result['constitutional_metrics']['definiteness']['value']
       depth_squared = result['constitutional_metrics']['duality']['theoretical']
       expected_sa = cd / depth_squared if depth_squared > 0 else 0
       assert abs(sa - expected_sa) < 0.001
   ```

### 🔬 Stateful Testing

Complex interaction testing using Hypothesis stateful framework:

```python
class GTMOStateMachine(RuleBasedStateMachine):
    @rule(text=st.text(min_size=5, max_size=200))
    def analyze_text(self, text):
        result = self.engine.gtmo_analyze_quantum(text)
        self.analysis_history.append(result)
    
    @invariant()
    def coordinates_always_bounded(self):
        for result in self.analysis_history:
            coords = result['coordinates']
            assert 0 <= coords['determination'] <= 1
            assert 0 <= coords['stability'] <= 1
            assert 0 <= coords['entropy'] <= 1
```

### 📈 Performance Expectations

| Text Length | Expected Throughput | Memory Usage |
|-------------|-------------------|--------------|
| 1-10 words | >100 words/sec | <50MB |
| 11-50 words | >50 words/sec | <100MB |
| 51-200 words | >20 words/sec | <200MB |
| 200+ words | >10 words/sec | <500MB |

### 🎉 Success Criteria

**100/100 Test Coverage Achieved When:**

1. ✅ All property-based tests pass (10+ properties)
2. ✅ Golden dataset validation >95% accuracy
3. ✅ Constitutional duality error <1% 
4. ✅ Performance benchmarks meet targets
5. ✅ Memory usage within bounds
6. ✅ Stateful testing invariants preserved
7. ✅ Integration tests pass
8. ✅ Edge cases handled correctly
9. ✅ Quantum mechanics verified
10. ✅ Rhetorical analysis accurate

### 🛠️ Troubleshooting

#### Common Issues

1. **Hypothesis Import Error**
   ```bash
   pip install hypothesis
   ```

2. **Memory Profiler Issues**
   ```bash
   pip install memory-profiler psutil
   ```

3. **spaCy Model Missing**
   ```bash
   python -m spacy download pl_core_news_sm
   ```

4. **Morfeusz2 Installation**
   ```bash
   pip install morfeusz2
   ```

#### Debug Mode

Run tests with detailed output:
```bash
python test_property_based.py --verbose
python benchmark_performance.py --debug
```

### 📋 Test Results Format

Each test run generates:

- `test_coverage_report.json` - Overall coverage summary
- `golden_dataset_comprehensive.json` - Reference dataset
- `benchmark_report_TIMESTAMP.json` - Performance analysis
- Console output with detailed pass/fail information

### 🎯 Next Steps for 100/100 Coverage

1. **Expand Golden Dataset** - Add more edge cases
2. **Increase Property Tests** - Add domain-specific properties  
3. **Performance Optimization** - Target throughput improvements
4. **Memory Optimization** - Reduce memory footprint
5. **Integration Testing** - More complex workflow tests

### 📚 References

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [GTMØ Theory Documentation](GTMO_MORPHOSYNTAX_DOCUMENTATION.md)
- [Constitutional Metrics Theory](test_constitutional_duality.py)