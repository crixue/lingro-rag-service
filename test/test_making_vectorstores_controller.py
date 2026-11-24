import os
import unittest

from configs.model_config import BASE_DIR


class TestDefaultMakingVSController(unittest.TestCase):

    default_config = {
        "mode": "single",
        "is_chinese_text": True,
        "cn_chunk_size": 300,
        "cn_chunk_overlap": 0,
        "en_chunk_size": 200,
        "en_chunk_overlap": 0,
        "separators": ["[SEP]", "\n\n", "\n", " ", ""]
    }

    def test_load_local_vs(self):
        from core.data_connections.retrievers.local_vectorstore_retriever.local_vector_store import DeprecatedLocalVectorStore
        local_vs = DeprecatedLocalVectorStore(folder_name=r"guest007/test_txt_file0", is_chinese_text=True, )
        query = "哪一个芯片制造厂商规模最大？"
        relevant_docs = local_vs.get_relevant_docs(query, search_kwargs={"k": 5})
        for doc in relevant_docs:
            print(doc)

    def test_txt_file(self):
        filepath = os.path.join(BASE_DIR, "samples", "0-words-cn.txt")
        making_vs_controller = DefaultMakingVSController(file_path=filepath, local_vs_folder_name=r"guest007/test_txt_file0",
                                                         **self.default_config)
        local_vs = making_vs_controller.add_documents_2_local_vs(need_clear_local_vs=True)
        query = "哪一个芯片制造厂商规模最大？"
        relevant_docs = local_vs.get_relevant_docs(query, search_kwargs={"k": 3})
        for doc in relevant_docs:
            print(doc)

    def test_txt_file_v2(self):
        filepath = os.path.join(BASE_DIR, "samples", "1-words-cn.txt")
        making_vs_controller = DefaultMakingVSController(file_path=filepath, local_vs_folder_name=r"guest007/test_txt_file0",
                                                         **self.default_config)
        local_vs = making_vs_controller.add_documents_2_local_vs(need_clear_local_vs=False)
        query = "哪一个芯片制造厂商规模最大？"
        relevant_docs = local_vs.get_relevant_documents(query, search_kwargs={"k": 3})
        for doc in relevant_docs:
            print(doc)

    def test_del_index_in_local_vector_store_v2(self):
        filepath = os.path.join(BASE_DIR, "samples", "1-words-cn.txt")
        doc_index_id_prefix = "80788ca1af604ec78363cde0f388ee6f"
        docsIndexCount = 6
        childDocsIndexCount = 49
        ids = [doc_index_id_prefix + "_" + str(i) for i in range(docsIndexCount)]
        sub_ids = [doc_index_id_prefix + "_sub_doc_" + str(i) for i in range(childDocsIndexCount)]
        making_vs_controller = DefaultMakingVSController(file_path=filepath, local_vs_folder_name=r"guest007/test_txt_file0",
                                                         **self.default_config)
        making_vs_controller.del_index_in_local_vector_store(ids=ids, sub_ids=sub_ids)
        local_vs = making_vs_controller.add_documents_2_local_vs(need_clear_local_vs=False)
        query = "哪一个芯片制造厂商规模最大？"
        relevant_docs = local_vs.get_relevant_documents(query, search_kwargs={"k": 3})
        for doc in relevant_docs:
            print(doc)

    def test_remove_all_vector_store(self):
        filepath = os.path.join(BASE_DIR, "samples", "0-words-cn.txt")
        making_vs_controller = DefaultMakingVSController(file_path=filepath, local_vs_folder_name=r"guest007/test_txt_file0",
                                                         **self.default_config)


    def test_html_file(self):
        filepath = os.path.join(BASE_DIR, "samples", "test-html-sample.html")
        making_vs_controller = DefaultMakingVSController(file_path=filepath, local_vs_folder_name=r"guest007/test_html_file0",
                                                         **self.default_config)
        local_vs = making_vs_controller.add_documents_2_local_vs(need_clear_local_vs=True)
        query = "哪一个芯片制造厂商规模最大？"
        relevant_docs = local_vs.get_relevant_docs(query, search_kwargs={"k": 3})
        for doc in relevant_docs:
            print(doc)

    def test_doc_file(self):
        # 0-words-sample.doc will convert to 0-words-sample.docx first
        filepath = os.path.join(BASE_DIR, "samples", "0-words-sample.doc")
        making_vs_controller = DefaultMakingVSController(file_path=filepath, local_vs_folder_name=r"guest007/test_doc_file0",
                                                         **self.default_config)
        local_vs = making_vs_controller.add_documents_2_local_vs(need_clear_local_vs=True)
        query = "药理学的课程目标是什么？"
        relevant_docs = local_vs.get_relevant_docs(query, search_kwargs={"k": 3})
        for doc in relevant_docs:
            print(doc)

    def test_docx_file(self):
        filepath = os.path.join(BASE_DIR, "samples", "0-test-docx-only-words.docx")
        making_vs_controller = DefaultMakingVSController(file_path=filepath, local_vs_folder_name=r"guest007/test_docx_file0",
                                                         **self.default_config)
        local_vs = making_vs_controller.add_documents_2_local_vs(need_clear_local_vs=True)
        query = "老师的教学成果如何？"
        relevant_docs = local_vs.get_relevant_docs(query, search_kwargs={"k": 3})
        for doc in relevant_docs:
            print(doc)

    def test_pdf_file(self):
        filepath = os.path.join(BASE_DIR, "samples", "0-words-with-pics.pdf")
        making_vs_controller = DefaultMakingVSController(file_path=filepath,
                                                         local_vs_folder_name=r"guest007/test_0-words-with-pics_pdf_file0",
                                                         **self.default_config)
        local_vs = making_vs_controller.add_documents_2_local_vs(need_clear_local_vs=True)
        query = "教学设计意图是什么？"  # 图片上的
        relevant_docs = local_vs.get_relevant_docs(query, search_kwargs={"k": 3})
        for doc in relevant_docs:
            print(doc)

    def test_json_file(self):
        '''
        How to use: https://python.langchain.com/docs/modules/data_connection/document_loaders/json
        '''
        # Define the metadata extraction function.
        def metadata_func(record: dict, metadata: dict) -> dict:
            metadata["sender_name"] = record.get("sender_name")
            metadata["timestamp_ms"] = record.get("timestamp_ms")

            return metadata

        filepath = os.path.join(BASE_DIR, "samples", "test-json-sample0.json")
        self.default_config.update({"is_chinese_text": False,
                                    "jq_schema": '.messages[].content',
                                    "metadata_func": metadata_func,
                                    "text_content": False
                                    })
        making_vs_controller = DefaultMakingVSController(file_path=filepath,
                                                         local_vs_folder_name=r"guest007/test-json-sample0",
                                                         **self.default_config)
        local_vs = making_vs_controller.add_documents_2_local_vs(need_clear_local_vs=True)
        query = "How much for blue bag?"  # 图片上的
        relevant_docs = local_vs.get_relevant_docs(query, search_kwargs={"k": 3})
        for doc in relevant_docs:
            print(doc)

    def test_SpeechRecognition_transcripts_json_file(self):
        '''
        local_vs_folder_name=r"guest007/test_txt_file0" have load before this test. And now we merge the transcripts json file to last local vector store.

        '''
        making_vs_controller = SpeechRecognitionMakingVSController(
                                                                   local_vs_folder_name=r"guest007/test_txt_file0",
                                                                   transcript_json_file_path=os.path.join(BASE_DIR, "samples", "test-mp4-sample.json"),
                                                                   **self.default_config)
        local_vs = making_vs_controller.add_documents_2_local_vs(need_clear_local_vs=False,
                                                                 metadata={"source": "test-mp4-sample.mp4"})
        query = "三星的芯片制造工艺是否比台积电更先进？"
        relevant_docs = local_vs.get_relevant_docs(query, search_kwargs={"k": 5})
        for doc in relevant_docs:
            print(doc)

    def test_plain_text(self):
        self.default_config.update({"is_chinese_text": True})
        making_vs_controller = ContentStringMakingVSController(local_vs_folder_name=r"guest007/test_plain_text0",
                                                               **self.default_config)
        content_str = "晶体管的平面工艺指的是在硅片表面依次制作源极、漏极和栅极的结构。在平面工艺中，栅极位于沟道上方，通过在栅极施加电压来控制沟道中载流子的浓度，从而控制源极与漏极之间的电流。这种结构在制程节点较大时表现稳定，工艺也相对简单，因此长期是集成电路制造的主流方案。随着制程不断缩小，栅极长度随之缩短，栅极与沟道之间的接触面积也在减小，栅极对沟道的控制能力逐渐变弱。当制程推进到二十纳米附近时，沟道中不同方向的电场相互干扰加剧，关断状态下的漏电流明显上升，这一现象通常被称为短沟道效应。短沟道效应不仅增加静态功耗，还会导致器件阈值电压难以控制，使芯片的良率与可靠性下降。为了解决短沟道效应，业界提出了把沟道做成三维结构的思路：让栅极从多个侧面包围沟道，以增大有效接触面积。鳍式场效应晶体管把沟道做成薄片状的鳍，栅极从三个侧面覆盖沟道，在相同占地面积下获得了更强的控制能力，也让制程得以继续缩小。进一步发展出的环绕栅极结构则让栅极从四个方向包围沟道，在更小的节点上保持较好的静电控制能力。从平面工艺到三维结构，核心目标始终是同一个：在缩小器件尺寸的同时，保持栅极对沟道的控制力，抑制漏电，并让器件的一致性满足量产要求。这一演进过程也说明，制程节点的推进并不只是把图形画得更小，还需要材料、结构、光刻和封装等多个环节协同改进。在数字电路设计中，工艺节点通常与标准单元库的密度和能效一起评估。同一节点下，不同厂商提供的标准单元高度、金属层数和供电方案可能不同，因此仅凭节点名称比较性能并不可靠。设计团队在选型时，一般会结合目标频率、功耗预算、芯片面积和量产时间综合判断，再通过综合与布局布线的结果验证是否满足约束。对模拟与电源管理类芯片而言，成熟节点往往更具成本优势，因为器件模型完善、IP 生态成熟、掩模成本更低，而且客户对工艺迁移的接受度较高。制造环节的良率管理同样依赖完整的量测与测试体系。光刻、刻蚀、薄膜沉积和离子注入等步骤都会引入偏差，工艺控制通过在线量测与统计过程控制来识别漂移，并在偏差超出阈值时触发调整。晶圆出厂前还要进行电性测试，记录每个器件的阈值电压、漏电流和跨导等参数，这些数据一方面用于筛选良品，另一方面也会回流到工艺端，帮助定位系统性缺陷。封装阶段则关注热阻、应力与互连可靠性，先进封装通过缩短互连距离来降低功耗并提升带宽。总体来看，工艺、设计与封装的协同决定了最终产品的性能与成本。单纯追求更小的节点，未必能带来更好的系统收益；在合适的节点上把设计、良率与供应链一起优化，往往才是更稳妥的工程选择。"
        local_vs = making_vs_controller.add_documents_2_local_vs(need_clear_local_vs=True,
                                                                 content=content_str)
        query = "能阐述下晶体管的平面工艺吗？"
        relevant_docs = local_vs.get_relevant_docs(query, search_kwargs={"k": 2})
        for doc in relevant_docs:
            print(doc)