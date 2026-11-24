from neo4j import GraphDatabase, AsyncGraphDatabase

from configs.model_config import NEO4J_CONFIG


class Neo4jGraphDBClient:
    def __init__(self,
                 uri = NEO4J_CONFIG["NEO4J_URI"],
                 user = NEO4J_CONFIG["NEO4J_USER"],
                 password = NEO4J_CONFIG["NEO4J_PASSWORD"]):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.async_driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    @property
    def graph_driver(self):
        return self.driver

    @property
    def async_graph_driver(self):
        return self.async_driver

    def close(self):
        self.driver.close()
        self.async_driver.close()

    def find_graphs_with_prefix(self, graph_name_prefix):
        query = "SHOW DATABASES"
        result = self.query(query, database="system")
        graph_names = [record["name"] for record in result if record["name"].startswith(graph_name_prefix)]
        return graph_names

    def query(self, query, database, parameters: dict = None):
        def run_query(tx):
            return tx.run(query, parameters).data()

        with self.driver.session(database=database) as session:
            result = session.execute_read(run_query)
            return result
