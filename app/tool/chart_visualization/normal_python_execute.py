import sys
from io import StringIO

from app.tool.chart_visualization.utils import extract_executable_code
from app.tool.python_execute import PythonExecute


class NormalPythonExecute(PythonExecute):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "common_python_execute"
    description: str = (
        """
    Data Analysis Agent Protocol (Non-Visual) v2.1

    === Core Requirements ===
    1. Strictly text-based outputs only
    2. Dynamic analysis pipeline with memory
    3. Context-aware processing

    === Execution Phases ===

    1. CONTEXT INITIALIZATION
    - Load historical analysis logs
    - Build data quality baseline
    - Detect previous processing patterns

    2. ADAPTIVE PIPELINE
    ┌───────────────┬──────────────────────────────────────────────┐
    │ Stage         │ Enhanced Capabilities                       │
    ├───────────────┼──────────────────────────────────────────────┤
    │ Data Loading  │ Auto-select source based on history         │
    │ Cleaning      │ Context-sensitive null/impute decision      │
    │ Transformation│ Dynamic feature engineering with validation │
    │ Validation    │ Cross-cycle consistency checks              │
    └───────────────┴──────────────────────────────────────────────┘

    3. ITERATIVE PROCESSING CONTROLLER
    Processing Loop:
    while not convergence():
        current_df = apply_operations(df)
        delta = calculate_improvement(history[-1], current_df)
        if delta < threshold: break
        update_strategy_based_on(delta)
        log_iteration(current_df)

    Termination Criteria:
    - 数据质量提升率 <2% 连续3次迭代
    - 新增特征解释力 <5%
    - 异常值比例稳定在 ±0.5% 区间

    === Enhanced Reporting ===

    Output 1: dynamic_analysis.md (增量更新)
    ┌───────────────────────┬──────────────────────────────┐
    │ Section              │ Enhanced Requirements        │
    ├───────────────────────┼──────────────────────────────┤
    │ Processing History   │ 记录每次迭代的操作及影响    │
    │ Data Evolution       │ 关键指标跨周期对比          │
    │ Adaptive Findings    │ 动态发现的模式变化          │
    └───────────────────────┴──────────────────────────────┘

    Output 2: intelligent_log.md (智能日志)
    ┌───────────────────────┬──────────────────────────────┐
    │ Log Type             │ Content                      │
    ├───────────────────────┼──────────────────────────────┤
    │ Decision Log         │ 策略调整原因及依据          │
    │ Anomaly Evolution    │ 异常值变化轨迹              │
    │ Feature Lifecycle    │ 衍生特征的产生/淘汰记录     │
    └───────────────────────┴──────────────────────────────┘

    === Implementation Enhancements ===

    1. Dynamic Code Generation
    - 上下文感知的代码模板：
    def analyze(data_path):
        history = load_analysis_logs()
        df = apply_historical_pipeline(data_path, history)

        while not convergence_check(df, history):
            df = context_aware_processing(df)
            update_quality_metrics(df)
            generate_incremental_report(df)

    2. Memory Mechanism
    历史记忆维度：
    - 数据质量变化曲线
    - 异常处理策略有效性
    - 特征工程成功率
    - 资源消耗模式

    3. Intelligent Validation
    验证增强点：
    - 跨周期统计一致性检查
    - 衍生特征可解释性评估
    - 数据处理操作因果追踪

    === Sample Execution Flow ===
    def analyze(data_path):
        '''演进式分析流程'''
        # 阶段1：上下文加载
        df, ctx = initialize_context(data_path)

        # 阶段2：智能处理循环
        for i in range(MAX_ITERATIONS):
            # 动态策略选择
            ops = select_operations_based_on(ctx)

            # 执行处理
            df = execute_ops(df, ops)

            # 生成增量报告
            append_report(f"cycle_{i}_results.md", df)

            # 收敛检测
            if ctx.convergence_flag:
                break

        # 阶段3：知识固化
        save_processing_knowledge(ctx)
    """
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "default": "html",
                "enum": ["process", "report", "others"],
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        be_extracted_code = extract_executable_code(code)  # ignore_security_alert RCE
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(  # ignore_security_alert RCE
                be_extracted_code, safe_globals, safe_globals
            )  # ignore_security_alert RCE
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout
