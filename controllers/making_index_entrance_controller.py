import asyncio
from concurrent.futures import ThreadPoolExecutor

from controllers.dispatch_handlers import BaseBuildingIndexStrategy, BuildingVectorStoreIndexStrategy, \
    BuildingHippoRAGGraphIndexStrategy, \
    BuildingElasticSearchIndexStrategy
from controllers.enums.build_knowledge_index_type_enum import BuildKnowledgeIndexTypeEnum
from controllers.vo.build_knowledge_index_request_vo import BuildKnowledgeIndexItemRequestVO, \
    DeleteKnowledgeBaseIndexRequestVO


class BuildingIndexEntranceController:
    def __init__(self, build_kg_index_type: BuildKnowledgeIndexTypeEnum = BuildKnowledgeIndexTypeEnum.LOCAL_VECTOR_STORE, **kwargs):
        '''
            Making vector store for file types.
            :param kwargs: Pls refer to BaseUnstructuredFileLoader
        '''
        self.build_kg_index_type = build_kg_index_type
        self.building_strategy = self._get_building_strategy(build_kg_index_type, **kwargs)

    @staticmethod
    def _get_building_strategy(build_kg_index_type: BuildKnowledgeIndexTypeEnum, **kwargs) -> BaseBuildingIndexStrategy:
        if build_kg_index_type == BuildKnowledgeIndexTypeEnum.HIPPO_KNOWLEDGE_GRAPH:
            return BuildingHippoRAGGraphIndexStrategy(**kwargs)
        elif build_kg_index_type == BuildKnowledgeIndexTypeEnum.ELASTICSEARCH:
            return BuildingElasticSearchIndexStrategy(**kwargs)
        else:
            return BuildingVectorStoreIndexStrategy(**kwargs)

    # def building_index(self, request: BuildKnowledgeIndexItemRequestVO):
    #     self.building_strategy.execute_with_preprocessing(request)

    def retry_failed_task(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs):
        self.building_strategy.retry_failed_task(request, **kwargs)

    def building_index(self, request: BuildKnowledgeIndexItemRequestVO):
        try:
            # 如果当前线程没有正在运行的事件循环，get_running_loop 会抛出 RuntimeError
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.building_strategy.execute_with_preprocessing_async(request))
        else:
            # 如果有正在运行的事件循环,则使用线程池在新线程中运行异步代码
            def _run():
                return asyncio.run(self.building_strategy.execute_with_preprocessing_async(request))

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run)
                return future.result()

    async def building_index_async(self, request: BuildKnowledgeIndexItemRequestVO):
        await self.building_strategy.execute_with_preprocessing_async(request)

    async def del_some_indies_async(self, request: DeleteKnowledgeBaseIndexRequestVO):
        await asyncio.to_thread(self.building_strategy.del_part_of_indies, request)

    async def del_version_of_kg_index(self, request: DeleteKnowledgeBaseIndexRequestVO):
        await asyncio.to_thread(self.building_strategy.del_version_of_indies_and_docs, request)

    async def del_workspace_async(self, request: DeleteKnowledgeBaseIndexRequestVO):
        await asyncio.to_thread(self.building_strategy.del_workspace, request)

    async def retry_failed_task_async(self, request: BuildKnowledgeIndexItemRequestVO, **kwargs):
        await asyncio.to_thread(self.building_strategy.retry_failed_task, request, **kwargs)