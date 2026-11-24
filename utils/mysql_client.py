import pymysql
import pymysql.cursors

from configs.model_config import MYSQL_CONFIG


class CustomMysqlClient:
    def __init__(self,):
        self.conn = pymysql.Connect(host=MYSQL_CONFIG["MYSQL_HOST"],
                        port=MYSQL_CONFIG["MYSQL_PORT"],
                        user=MYSQL_CONFIG["MYSQL_USER"],
                        password=MYSQL_CONFIG["MYSQL_PASSWORD"],
                        database=MYSQL_CONFIG["MYSQL_DBNAME"],
                        charset=MYSQL_CONFIG["MYSQL_CHARSET"],
                        use_unicode=True,
                        cursorclass=pymysql.cursors.DictCursor,
                                    )
        self.cursor = self.conn.cursor()

    def get_cursor(self):
        return self.cursor

    def close(self):
        self.cursor.close()
        self.conn.close()
