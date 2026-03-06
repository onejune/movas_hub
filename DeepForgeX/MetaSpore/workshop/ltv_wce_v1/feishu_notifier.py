# feishu_notifier.py
'''
模型验证结果发送飞书消息通用模块
使用流程:
    1. 自建一个飞书群，添加机器人，并获取群的 WebHook URL；
    2. 拿WebHook URL跟李武申请token和 topic；
    3. 替换代码中的以上 3 个变量；
'''
import json
import requests
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

class FeishuNotifier:
    # 飞书 WebHook URL，跟飞书群绑定, token和 topic 需要找管理员申请
    ################### private config #################
    _WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/a6b95cf6-1715-4c8e-b96d-bbc8928a19bf"
    _TOKEN = "0cdb867f-2837-428f-a0f8-64e64586cbdd"
    _TOPIC = "dsp_alert3"
    ####################################################

    @staticmethod
    #纯文本发送飞书消息
    def send_text(text: str, timeout: int = 10) -> Dict:
        if not text or not isinstance(text, str):
            raise ValueError("Message text must be a non-empty string")

        now = datetime.now()
        formatted_dtm = now.strftime("%Y-%m-%d %H:%M:%S")

        payload = {
            "msg_type": "text",
            "content": {"text": formatted_dtm + '\n' + text}
        }

        try:
            response = requests.post(
                FeishuNotifier._WEBHOOK_URL,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to send message to Feishu: {e}")

    @staticmethod
    #以表格图片的形式发送飞书消息
    def send_html(html_content: str, subject: str = "HTML Message") -> Dict:
        """
        通过自定义 API 发送 HTML 内容到飞书
        """
        url = "http://api.ymnotify.dy/send"
        headers = {
            "Content-Type": "application/json",
            "token":FeishuNotifier. _TOKEN
        }
        data = {
            "client": "zmaticoo_dsp",
            "topic": FeishuNotifier._TOPIC,  # 写死的 topic
            "subject": subject,
            "alliance_id": "2",
            "content": html_content
        }
        try:
            response = requests.post(url=url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = {"status": "success", "data": response.json()}
            print(result)
            return result
        except requests.exceptions.RequestException as e:
            result = {"status": "failed", "error": str(e)}
            print(result)
            return result

    @staticmethod
    def dataframe_to_html_table(df: pd.DataFrame, title: str = "Table", text: str = "") -> str:
        # 创建 HTML 表格
        html = f"""
        <div style="text-align: center; margin: 10px 0;">
            <!-- 文本内容区域 -->
            {f'<div style="text-align: left; margin: 15px 0; line-height: 1.6; white-space: pre-line;">{text}</div>' if text else ''}
            
            <!-- 标题区域 -->
            <h4 style="margin: 15px 0; text-align: center; color: #333;">{title}</h4>
        </div>
        """
        
        html += f"""
        <div style="overflow-x: auto;">
            <table style="
                border-collapse: collapse;
                width: 100%;
                border: 1px solid #ddd;
                font-family: Arial, sans-serif;
                font-size: 12px;
                margin: 15px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <thead>
                    <tr style="background-color: #4CAF50; color: white; font-weight: bold;">
        """
        
        # 添加表头
        for col in df.columns:
            html += f'<th style="border: 1px solid #ddd; padding: 10px; text-align: center; min-width: 80px;">{col}</th>'
        
        html += "</tr></thead><tbody>"
        
        # 添加数据行
        for idx, row in df.iterrows():
            # 交替行颜色以提高可读性
            row_style = "background-color: #f9f9f9;" if idx % 2 == 0 else "background-color: #ffffff;"
            html += f"<tr style='{row_style}'>"
            for cell in row:
                # 处理数字格式，保留适当的小数位
                if isinstance(cell, (int, float)):
                    if isinstance(cell, float):
                        # 对于小数，保留4位小数但去除尾随零
                        formatted_cell = f"{cell:.4f}".rstrip('0').rstrip('.')
                        cell = formatted_cell if formatted_cell != '' else '0'
                    else:
                        cell = str(cell)
                html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center; vertical-align: top;">{cell}</td>'
            html += "</tr>"
        
        html += """
            </tbody>
        </table>
        </div>
        """
        return html

    @staticmethod
    #对外接口，发送表格图片消息, title: 表格标题, subject: 消息主题, text: 消息内容
    def send_dataframe_html(df: pd.DataFrame, title: str = "Table", subject: str = "Notice", text: str = "") -> Dict:
        html_table = FeishuNotifier.dataframe_to_html_table(df, title, text)
        return FeishuNotifier.send_html(html_table, subject)

    @staticmethod
    def send_evaluation_results_html(sorted_results: List, test_date_str_formatted: str, 
                                   model_name: str) -> Dict:
        if not sorted_results:
            html_content = "<h3>模型评估结果</h3><p>暂无评估数据</p>"
            return FeishuNotifier.send_html(html_content, f"{model_name} 模型评估结果")
        
        # 构建表格数据
        table_data = []
        tag = f"val-{model_name}-{test_date_str_formatted}"
        
        for result in sorted_results:
            row = {
                "DATE": tag,
                "Key1": getattr(result, 'key1', ''),
                "Key2": getattr(result, 'key2', ''),
                "AUC": round(getattr(result, 'auc', 0), 4),
                "PCOC": round(getattr(result, 'pcoc', 0), 4),
                "LOSS": round(getattr(result, 'loss', 0), 4),
                "POS": getattr(result, 'pos', 0),
                "NEG": getattr(result, 'neg', 0),
                "IVR": round(getattr(result, 'ivr', 0), 4)
            }
            table_data.append(row)
        
        # 创建 DataFrame
        df = pd.DataFrame(table_data)
        
        # 发送 HTML 表格
        return FeishuNotifier.send_dataframe_html(
            df, 
            f"模型评估结果 - {model_name}", 
            f"{model_name} 模型评估结果"
        )

    # 快捷别名（可选）
    @staticmethod
    def notify(text: str) -> Dict:
        """快捷发送方法"""
        return FeishuNotifier.send_text(text)

if __name__ == "__main__":
    # 测试 HTML 表格发送
    try:
        import pandas as pd
        test_df = pd.DataFrame({
            'Model': ['DeepFM', 'WideDeep', 'FFM', 'DCN'],
            'AUC': [0.85, 0.83, 0.82, 0.84],
            'PCOC': [0.92, 0.90, 0.88, 0.91],
            'LogLoss': [0.25, 0.27, 0.28, 0.26],
            'POS': [1000, 950, 980, 970],
            'NEG': [9000, 9050, 9020, 9030],
            'IVR': [0.1, 0.095, 0.098, 0.097]
        })
        
        result = FeishuNotifier.send_dataframe_html(test_df,"validation result", "model_text_v2", "这是一个模型评估结果消息发送的测试消息\n换一行")
        print("HTML表格发送成功:", result)
    except Exception as e:
        print("HTML表格发送失败:", e)
    