# RAG 评测

`rag_checker/` 是把 [RAGChecker](https://github.com/amazon-science/RAGChecker) 的论文指标落成可调用服务的实现，另加了一个轻量的 LLM-as-Judge。

两部分评测回答的是不同问题：

| 评测器 | 回答的问题 | 依赖 |
|---|---|---|
| `RAGChecker` | 检索准不准、回答有没有依据、有没有幻觉 | claim 抽取 + claim 校验（`refchecker`） |
| `CustomChecker` | 这条回答整体好不好用 | 一次带评分标准的 LLM 调用 |

---

## 1. 指标分组

指标定义见 `rag_checker/metrics.py`，计算见 `rag_checker/computation.py`。

### 整体指标（overall_metrics）

| 指标 | 含义 |
|---|---|
| `sentence_precision` | 回答中每个句子与标准答案的匹配精度 |
| `precision` | 回答的 claim 有多少被标准答案覆盖 |
| `recall` | 标准答案的 claim 有多少被回答覆盖 |
| `f1` | precision / recall 的调和平均 |

### 检索指标（retriever_metrics）

| 指标 | 含义 |
|---|---|
| `claim_recall` | 回答所需 claim 中，检索文档实际覆盖的比例 |
| `context_precision` | 检索文档中，真正与标准答案相关的比例 |

### 生成指标（generator_metrics）

| 指标 | 含义 |
|---|---|
| `context_utilization` | 检索到的正确信息被用上了多少 |
| `noise_sensitivity_in_relevant` | 相关噪声导致的错误率 |
| `noise_sensitivity_in_irrelevant` | 不相关噪声导致的错误率 |
| `hallucination` | 检索内容不支持且标准答案也不支持的 claim 比例 |
| `self_knowledge` | 检索不支持但标准答案支持的 claim 比例（模型自身知识补全） |
| `faithfulness` | 回答中能被检索内容支持的 claim 比例 |

`METRIC_REQUIREMENTS` 声明每个指标依赖哪些中间结果（`answer2response`、`retrieved2answer` 等），因此可以只跑子集，避免重复调用模型。

---

## 2. 输入数据格式

`RAGResults.from_json()` 接受如下结构（`rag_checker/container.py`）：

```json
{
  "results": [
    {
      "query_id": "q1",
      "query": "示例半导体的 28nm 工艺是否量产？",
      "gt_answer": "已量产，良率稳定在九成以上。",
      "response": "已量产。",
      "retrieved_context": [
        {"doc_id": "doc-1", "text": "示例半导体宣布 28nm 工艺实现量产。"}
      ]
    }
  ],
  "metrics": {
    "overall_metrics": {},
    "retriever_metrics": {},
    "generator_metrics": {}
  }
}
```

其余字段（`response_claims`、`answer2response` 等）由评测过程填充，输入时留空即可。

---

## 3. 命令行用法

```bash
python -m rag_checker.cli \
  --input_path  ./eval/sample_input.json \
  --output_path ./eval/sample_output.json \
  --extractor_name   Qwen/Qwen3-32B \
  --extractor_api_base http://127.0.0.1:8000/v1 \
  --checker_name     Qwen/Qwen3-32B \
  --checker_api_base http://127.0.0.1:8001/v1 \
  --metrics all_metrics \
  --joint_check_num 5
```

| 参数 | 说明 |
|---|---|
| `--extractor_name` / `--extractor_api_base` | claim 抽取模型及其 OpenAI 兼容入口 |
| `--checker_name` / `--checker_api_base` | claim 校验模型（可与抽取模型不同） |
| `--metrics` | 指标名或分组名，默认 `all_metrics` |
| `--joint_check_num` | 一次校验合并的 claim 数，越大越省调用、越容易误判 |
| `--disable_joint_check` | 逐条校验，成本高但更稳 |

输出会把每个样本的中间结果与聚合指标写回文件，便于定位是哪类错误。

---

## 4. 服务内调用

`POST /rag-checker/v1/evaluate` 由 `controllers/rag_checker_controller.py` 处理：

1. 读取/初始化工作区下的 `rag_test_data.json`；
2. 用请求里的 `retriever_config` 对每个问题跑一次真实检索；
3. 用 `used_model_config` 指定的模型生成回答；
4. 调用 `RAGEvaluator` 计算指标；
5. 结果写入工作区缓存目录，并回调 Java 服务 `notice-rag-check-result`。

这意味着评测跑的是**线上同一条检索链路**，而不是离线拼装的数据，能反映真实召回质量。

---

## 5. 自定义 LLM Judge

`CustomChecker.sentence_semantic_check` 用一条带评分标准的 Prompt（`prompt_definitions/builtin/lang/**/semantic_evaluator_prompt.json`）让模型对回答打分，并解析加权总分：

| 维度 | 说明 |
|---|---|
| content accuracy | 事实是否正确 |
| completeness | 是否覆盖必要信息 |
| relevance | 是否切题 |
| conciseness | 是否冗余 |

解析失败或模型输出不合规时返回 `N/A` 而不是 0，避免把"评测失败"误当成"回答很差"。

---

## 6. 使用建议

- **先建基线再调参**：固定一组问题，记录 precision / claim_recall / faithfulness，改检索策略后必须复跑；
- **区分检索与生成**：`claim_recall` 下降说明召回有问题，`faithfulness` 下降说明模型开始编造，两者要分开定位；
- **控制评测成本**：claim 抽取与校验都是 LLM 调用，样本量建议 30–50 条起步，`joint_check_num` 取 5；
- **注意答案长度偏差**：`precision` 对短回答友好，`recall` 对长回答友好，看 `f1` 更容易比较不同版本。

> 本仓库不包含任何真实评测数据集与线上指标。上面的输入示例为合成数据。
