from mcp.server.fastmcp import FastMCP
import requests
from openai import OpenAI
from tavily import TavilyClient
import os
import logging
import pandas as pd
from prompts import *

mcp = FastMCP("search")

# 配置 OpenAI
base_url = "https://openrouter.ai/api/v1"
api_key = 'aaa'  
model_name = 'deepseek/deepseek-chat:free'

# 配置 Tavily
tavily_api_key = 'aaa'  
tavily = TavilyClient(api_key=tavily_api_key)

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 创建文件处理器
file_handler = logging.FileHandler('test.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

def generate_query(query, stream=False):
    """生成搜索查询"""
    try:
        prompt = """You are an expert research assistant. Given the user's query, generate up to four distinct, precise search queries that would help gather comprehensive information on the topic.
        Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."""
            
        response = client.chat.completions.create(
            model=model_name,
            messages = [
                {"role": "system", "content": "You are a helpful and precise research assistant."},
                {"role": "user", "content": f"User Query: {query}\n\n{prompt}"}
            ]
        )
        
        response_text = response.choices[0].message.content
        
        # 尝试提取和清理列表部分
        if '[' in response_text and ']' in response_text:
            list_part = response_text[response_text.find('['):response_text.rfind(']')+1]
            return list_part
        else:
            # 如果找不到列表，返回默认查询
            logger.warning(f"No list found in LLM response: {response_text}")
            return f"['{query}']"
    except Exception as e:
        logger.error(f"Error in generate_query: {e}")
        return f"['{query}']"  # 默认使用原始查询

def web_search(query: str) -> list:
    """使用 Tavily API 进行网络搜索，返回相关链接列表"""
    links = []
    try:
        logger.info(f"使用 Tavily 搜索: {query}")
        
        # 进行搜索，获取结果
        search_response = tavily.search(
            query=query,
            search_depth="basic",  # 可以是 "basic" 或 "advanced"
            max_results=5,         # 获取的最大结果数量
            include_domains=[],    # 可选：仅包括特定域名
            exclude_domains=[]     # 可选：排除特定域名
        )
        
        # 从响应中提取链接
        if "results" in search_response:
            for result in search_response["results"]:
                if "url" in result:
                    links.append(result["url"])
            logger.info(f"Tavily 搜索返回了 {len(links)} 个链接")
        else:
            logger.warning(f"Tavily 没有返回结果，响应: {search_response}")
            
    except Exception as e:
        logger.error(f"Tavily 搜索出错: {str(e)}")
        # 添加一些备用链接，以防搜索失败
        links = ["https://example.com/search-failed"]
        
    return links

def if_useful(query: str, page_text: str):
    """判断页面内容是否有用"""
    prompt = """You are a critical research evaluator. Given the user's query and the content of a webpage, determine if the webpage contains information relevant and useful for addressing the query.
    Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. Do not include any extra text."""
    
    response = client.chat.completions.create(
        model=model_name,
        messages = [
            {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
            {"role": "user", "content": f"User Query: {query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
        ]
    )
    
    response = response.choices[0].message.content
    
    if response:
        answer = response.strip()
        if answer in ["Yes", "No"]:
            return answer
        else:
            # Fallback: try to extract Yes/No from the response.
            if "Yes" in answer:
                return "Yes"
            elif "No" in answer:
                return "No"
    return "No"

def extract_relevant_context(query, search_query, page_text):
    """提取页面中与查询相关的内容"""
    prompt = """You are an expert information extractor. Given the user's query, the search query that led to this page, and the webpage content, extract all pieces of information that are relevant to answering the user's query.
    Return only the relevant context as plain text without commentary."""
    
    response = client.chat.completions.create(
        model=model_name,
        messages = [
            {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
            {"role": "user", "content": f"User Query: {query}\nSearch Query: {search_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
        ]
    )
    
    response = response.choices[0].message.content
    if response:
        return response.strip()
    return ""

def get_new_search_queries(user_query, previous_search_queries, all_contexts):
    """根据已有结果，决定是否需要进一步搜索"""
    context_combined = "\n".join(all_contexts)
    prompt = """You are an analytical research assistant. Based on the original query, the search queries performed so far, and the extracted contexts from webpages, determine if further research is needed.
    If further research is needed, provide up to four new search queries as a Python list (for example, ['new query1', 'new query2']). If you believe no further research is needed, respond with exactly .
    Output only a Python list or the token  without any additional text."""
    
    response = client.chat.completions.create(
        model=model_name,
        messages = [
            {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
            {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}"}
        ]
    )
    
    response = response.choices[0].message.content
    if response:
        cleaned = response.strip()
        if cleaned == "":
            return ""
        try:
            new_queries = eval(cleaned)
            if isinstance(new_queries, list):
                return new_queries
            else:
                logger.info(f"LLM did not return a list for new search queries. Response: {response}")
                return []
        except Exception as e:
            logger.error(f"Error parsing new search queries:{e}, Response:{response}")
            return []
    return []

def fetch_webpage_text(url):
    """获取网页内容"""
    try:
        # 从URL提取域名或路径作为查询参数
        import re
        domain = re.sub(r'https?://', '', url)
        domain = domain.split('/')[0]  # 获取域名部分
        query = f"website information {domain}"
        
        # 使用非空查询和URL域名
        logger.info(f"Fetching content with query: '{query}' and URL: {url}")
        search_response = tavily.search(
            query=query,
            search_depth="basic",
            include_domains=[url],  # 仅包含这个特定URL
            max_results=1
        )
        
        if "results" in search_response and search_response["results"]:
            result = search_response["results"][0]
            if "content" in result:
                return result["content"]
        
        # 如果 Tavily 没有返回内容，直接返回空字符串
        logger.warning(f"Tavily did not return content for {url}")
        return ""
    except Exception as e:
        logger.error(f"Error fetching webpage text from Tavily: {e}")
        return ""
    
def process_link(link, query, search_query):
    """处理单个链接：获取内容、判断是否有用、提取相关内容"""
    logger.info(f"Fetching content from: {link}")
    page_text = fetch_webpage_text(link)
    if not page_text:
        return None
    usefulness = if_useful(query, page_text)
    logger.info(f"Page usefulness for {link}: {usefulness}")
    if usefulness == "Yes":
        context = extract_relevant_context(query, search_query, page_text)
        if context:
            logger.info(f"Extracted context from {link} (first 200 chars): {context[:200]}")
            return context
    return None

def get_images_description(image_url):
    """使用多模态模型为图片生成描述"""
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen2.5-vl-32b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "使用一句话描述图片的内容"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting image description: {e}")
        return "无法获取图片描述"

@mcp.tool()
def search(query: str) -> str:
    """互联网搜索"""
    try:
        iteration_limit = 3
        iteration = 0
        aggregated_contexts = []  
        all_search_queries = []   
        
        # 生成搜索查询
        try:
            new_search_queries = eval(generate_query(query))
            if not isinstance(new_search_queries, list) or len(new_search_queries) == 0:
                logger.warning(f"LLM did not return a valid query list, using original query")
                new_search_queries = [query]
        except Exception as e:
            logger.error(f"Error evaluating search queries: {e}")
            new_search_queries = [query]  # 使用原始查询作为回退
            
        all_search_queries.extend(new_search_queries)
        
        while iteration < iteration_limit:
            logger.info(f"\n=== Iteration {iteration + 1} ===")
            iteration_contexts = []
            
            # 为每个查询执行搜索
            search_results = []
            for q in new_search_queries:
                links = web_search(q)
                search_results.append(links)

            # 收集唯一链接及其对应的搜索查询
            unique_links = {}
            for idx, links in enumerate(search_results):
                query = new_search_queries[idx]
                for link in links:
                    if link not in unique_links:
                        unique_links[link] = query

            logger.info(f"Aggregated {len(unique_links)} unique links from this iteration.")

            # 处理每个链接：获取内容、判断是否有用、提取相关内容
            link_results = []
            for link in unique_links:
                result = process_link(link, query, unique_links[link])
                if result:
                    link_results.append(result)
            
            # 收集有用的上下文
            for res in link_results:
                if res:
                    iteration_contexts.append(res)

            if iteration_contexts:
                aggregated_contexts.extend(iteration_contexts)
            else:
                logger.info("No useful contexts were found in this iteration.")

            # 决定是否需要更多搜索
            new_search_queries = get_new_search_queries(query, all_search_queries, aggregated_contexts)
            if new_search_queries == "":
                logger.info("LLM indicated that no further research is needed.")
                break
            elif new_search_queries:
                logger.info(f"LLM provided new search queries:{new_search_queries}")
                all_search_queries.extend(new_search_queries)
            else:
                logger.info("LLM did not provide any new search queries. Ending the loop.")
                break

            iteration += 1
            
        if not aggregated_contexts:
            return "搜索未找到相关信息。请尝试修改您的查询或提供更多细节。"
            
        return '\n\n'.join(aggregated_contexts)
    except Exception as e:
        logger.error(f"Error in search function: {str(e)}")
        return f"搜索时发生错误: {str(e)}"

@mcp.tool()
def get_images(query: str) -> str:
    '''获取图片链接和描述'''
    logger.info(f"搜索图片: {query}")
    
    result = {}
    try:
        # 使用 Tavily 搜索相关内容，可能包含图片
        search_response = tavily.search(
            query=f"{query} 图片",
            search_depth="advanced",
            max_results=5
        )
        
        # 从 Tavily 结果中提取图片链接
        image_links = []
        if "results" in search_response:
            for item in search_response["results"]:
                # 如果结果中包含图片链接
                if "image_url" in item and item["image_url"]:
                    image_links.append(item["image_url"])
                # 备用：从内容中提取可能的图片 URL
                elif "url" in item and any(ext in item["url"].lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                    image_links.append(item["url"])
        
        # 如果没有找到图片，使用搜索结果的前两个 URL 作为替代
        if not image_links and "results" in search_response:
            for item in search_response["results"][:2]:
                if "url" in item:
                    image_links.append(item["url"])
        
        # 最多处理2张图片
        for img_src in image_links[:2]:
            logger.info(f"获取图片描述: {img_src}")
            description = get_images_description(img_src)
            logger.info(f"图片 {img_src} 描述: {description}")
            result[img_src] = description
        
        if not result:
            # 如果没有找到任何图片，添加一个消息
            logger.warning(f"未找到与 '{query}' 相关的图片")
            result["no_images"] = f"未找到与 '{query}' 相关的图片"
            
    except Exception as e:
        logger.error(f"获取图片时出错: {str(e)}")
        result["error"] = f"获取图片时发生错误: {str(e)}"
        
    return result

if __name__ == "__main__":
    mcp.run()