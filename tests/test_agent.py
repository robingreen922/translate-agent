import json#导入json模块，用于将字符串转换为字典
import os#导入os模块，用于操作文件和目录
from unittest.mock import patch#导入patch类，用于模拟函数调用

import openai#导入openai模块
import pytest#导入pytest模块，用于测试
import openai#导入openai模块
from openai import OpenAI#导入OpenAI模块,用于调用openai的接口
from dotenv import load_dotenv#导入load_dotenv函数，用于加载环境变量

# from translation_agent.utils import find_sentence_starts
from translation_agent.utils import get_completion#导入get_completion函数，用于获取文本翻译
from translation_agent.utils import num_tokens_in_string#导入num_tokens_in_string函数，用于计算字符串中的单词数
from translation_agent.utils import one_chunk_improve_translation#导入one_chunk_improve_translation函数，用于改进翻译
from translation_agent.utils import one_chunk_initial_translation#导入one_chunk_initial_translation函数，用于初始翻译
from translation_agent.utils import one_chunk_reflect_on_translation#导入one_chunk_reflect_on_translation函数，用于反思翻译
from translation_agent.utils import one_chunk_translate_text#导入one_chunk_translate_text函数，用于翻译文本


load_dotenv()#读取环境变量
API_KEY = os.getenv("OPENAI_API_KEY")#获取openai的API_KEY
API_URL = os.getenv("OPENAI_API_URL")#获取openai的API_URL
client = OpenAI(
    api_key = API_KEY,#设置openai的API_KEY
    base_url = API_URL#设置openai的API_URL
)
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))#创建openai客户端


def test_get_completion_json_mode_api_call():#测试get_completion函数
    # Set up the test data
    prompt = "What is the capital of France in json?"#提示信息，获取法国首都
    system_message = "You are a helpful assistant."#系统消息，你是一个有用的助手
    model = "gpt-3.5-turbo"#模型，gpt-4-turbo
    temperature = 0.3#温度，0.3
    json_mode = True#json模式，True

    # Call the function with JSON_mode=True
    result = get_completion(#调用get_completion函数
        prompt, system_message, model, temperature, json_mode#提示信息，系统消息，模型，温度，json模式
    )
    #打印结果result的结果是
    print("result："+result)
    #测试结果是否为空，如果为空则抛出异常
    assert result is not None

    #测试结果是否是字典类型，如果不是则抛出异常
    assert isinstance(json.loads(result), dict)


def test_get_completion_non_json_mode_api_call():#测试get_completion函数,非json模式
    # Set up the test data
    prompt = "What is the capital of France?"#提示信息，获取法国首都
    system_message = "You are a helpful assistant."#系统消息，你是一个有用的助手
    model = "gpt-3.5-turbo"#模型，gpt-4-turbo
    temperature = 0.3#温度，0.3
    json_mode = False#json模式，False

    # Call the function with JSON_mode=False
    result = get_completion(#调用get_completion函数
        prompt, system_message, model, temperature, json_mode#提示信息，系统消息，模型，温度，json模式
    )
    #打印结果result的结果是
    print("result："+result)
    # Assert that the result is not None
    assert result is not None

    # Assert that the result has the expected response format
    assert isinstance(result, str)


def test_one_chunk_initial_translation():#测试one_chunk_initial_translation函数
    # Define test data
    source_lang = "English"#源语言，英语
    target_lang = "Chinese"#目标语言，中文
    source_text = "Hello, how are you?"#源文本，你好，你好吗？
    expected_translation = "你好，最近过得怎么样？"#预期翻译结果，你好，你好吗？

    # Mock the get_completion_content function
    with patch(#模拟get_completion_content函数
        "translation_agent.utils.get_completion"#函数路径
    ) as mock_get_completion:#mock_get_completion,模拟get_completion函数
        mock_get_completion.return_value = expected_translation#返回值为预期翻译结果

        # Call the function with test data
        translation = one_chunk_initial_translation(#调用one_chunk_initial_translation函数
            source_lang, target_lang, source_text#源语言，目标语言，源文本
        )
        #打印翻译结果translation
        print("translation："+translation)
        # Assert the expected translation is returned
        assert translation == expected_translation

        # Assert the get_completion_content function was called with the correct arguments
        # 断言get_completion_content函数被正确调用
        expected_system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
        # 断言get_completion_content函数被正确调用
        expected_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""
        # 断言get_completion_content函数被正确调用
        mock_get_completion.assert_called_once_with(
            expected_prompt, system_message=expected_system_message
        )


def test_one_chunk_reflect_on_translation():#测试one_chunk_reflect_on_translation函数
    # Define test data
    source_lang = "English"#源语言，英语
    target_lang = "Chinese"#目标语言，中文
    country = "China"#国家，中国
    source_text = "Hello, how are you?"#源文本，这是一个示例源文本
    translation_1 = "你好，最近过得怎么样？"#翻译结果，这是一个例子

    # Define the expected reflection
    # 定义预期反思，这是一个示例源文本
    expected_reflection = "The translation is generally accurate and conveys the meaning of the source text well. However, here are a few suggestions for improvement:\n\n1. Consider using '你好，近来可好？' instead of '你好，最近过得怎么样？' for a more natural translation of 'Hello, how are you?'.\n2. To improve fluency, you might also say '你最近还好吗？' which is a commonly used phrase.\n3. Make sure to match the tone and formality of the original text; '近来可好' is slightly more formal and might be appropriate in certain contexts."

    # Mock the get_completion_content function
    with patch(#模拟get_completion_content函数
        "translation_agent.utils.get_completion"#函数路径
    ) as mock_get_completion:#mock_get_completion,模拟get_completion函数
        mock_get_completion.return_value = expected_reflection#返回值为预期反思

        # Call the function with test data
        reflection = one_chunk_reflect_on_translation(#调用one_chunk_reflect_on_translation函数
            source_lang, target_lang, source_text, translation_1, country#源语言，目标语言，源文本，翻译结果，国家
        )
        #打印反思reflection
        print("reflection："+reflection)
        # Assert that the reflection matches the expected reflection
        assert reflection == expected_reflection

        # Assert that the get_completion_content function was called with the correct arguments
        # 断言get_completion_content函数被正确调用
        expected_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""
        # 断言get_completion_content函数被正确调用
        expected_system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."
        # 打印生成的提示和系统消息
        print("expected_prompt: ", expected_prompt)
        print("expected_system_message: ", expected_system_message)
        # 验证 get_completion 函数被正确调用
        mock_get_completion.assert_called_once_with(
            expected_prompt, system_message=expected_system_message#断言get_completion_content函数被正确调用
        )

#装饰器，用于提供测试数据
@pytest.fixture
def example_data():#定义测试数据
    return {#返回字典
        "source_lang": "English",#源语言，英语
        "target_lang": "Chinese",#目标语言，中文
        "source_text": "Hello, how are you?",#源文本，这是一个示例源文本
        "translation_1": "你好，最近过得怎么样?",#翻译结果，这是一个例子
        "reflection": "The translation is accurate but could be more fluent.",#反思，翻译准确但可以更流畅
    }


@patch("translation_agent.utils.get_completion")#模拟get_completion函数
def test_one_chunk_improve_translation(mock_get_completion, example_data):#测试one_chunk_improve_translation函数
    #模拟get_completion_content函数的返回值
    mock_get_completion.return_value = (
        "你好，最近过得怎么样?"#返回值，这是一个改进的示例翻译
    )

    # Call the function with the example data
    result = one_chunk_improve_translation(
        example_data["source_lang"],#源语言
        example_data["target_lang"],#目标语言
        example_data["source_text"],#源文本
        example_data["translation_1"],#翻译结果
        example_data["reflection"],#反思
    )
    #打印翻译结果result
    print("result："+result)
    # Assert that the function returns the expected translation
    assert result == "你好，最近过得怎么样?"

    # Assert that get_completion was called with the expected arguments
    # 断言get_completion被调用时传入了预期参数
    expected_prompt = f"""Your task is to carefully read, then edit, a translation from {example_data["source_lang"]} to {example_data["target_lang"]}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{example_data["source_text"]}
</SOURCE_TEXT>

<TRANSLATION>
{example_data["translation_1"]}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{example_data["reflection"]}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying Chinese grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""
    expected_system_message = f"You are an expert linguist, specializing in translation editing from English to Chinese."
    mock_get_completion.assert_called_once_with(
        expected_prompt, expected_system_message
    )


def test_one_chunk_translate_text(mocker):#测试one_chunk_translate_text函数
    # Define test data
    source_lang = "English"#源语言，英语
    target_lang = "Chinese"#目标语言，中文
    country = "China"#国家，中国
    source_text = "Hello, how are you?"#源文本，你好，最近过得怎么样？
    translation_1 = "你好，最近过得怎么样?"#翻译结果，你好，最近过得怎么样？
    reflection = "The translation looks good, but it could be more formal."#反思，翻译看起来不错，但可以更正式
    translation2 = "您好，最近咋样?"

    # Mock the helper functions
    mock_initial_translation = mocker.patch(#模拟initial_translation函数
        "translation_agent.utils.one_chunk_initial_translation",
        return_value=translation_1
    )
    mock_reflect_on_translation = mocker.patch(#模拟reflect_on_translation函数
        "translation_agent.utils.one_chunk_reflect_on_translation",
        return_value=reflection,
    )
    mock_improve_translation = mocker.patch(#模拟improve_translation函数
        "translation_agent.utils.one_chunk_improve_translation",
        return_value=translation2,
    )
    # Call the function being tested
    result = one_chunk_translate_text(#调用one_chunk_translate_text函数
        source_lang, target_lang, source_text, country#源语言，目标语言，源文本，国家
    )
    #打印翻译结果result
    print("result："+result)
    # Assert the expected result
    assert result == translation2#断言结果与预期结果匹配

    # Assert that the helper functions were called with the correct arguments
    mock_initial_translation.assert_called_once_with(#断言initial_translation函数被正确调用
        source_lang, target_lang, source_text#源语言，目标语言，源文本
    )
    mock_reflect_on_translation.assert_called_once_with(#断言reflect_on_translation函数被正确调用
        source_lang, target_lang, source_text, translation_1, country#源语言，目标语言，源文本，翻译结果，国家
    )
    mock_improve_translation.assert_called_once_with(#断言improve_translation函数被正确调用
        source_lang, target_lang, source_text, translation_1, reflection#源语言，目标语言，源文本，翻译结果，反思
    )


def test_num_tokens_in_string():#测试num_tokens_in_string函数
    # Test case 1: Empty string
    assert num_tokens_in_string("") == 0#断言空字符串的单词数为0

    # Test case 2: Simple string
    assert num_tokens_in_string("Hello, world!") == 4#断言简单字符串的单词数为4

    # Test case 3: String with special characters
    assert (
        num_tokens_in_string(#断言字符串中的特殊字符的单词数
            "This is a test string with special characters: !@#$%^&*()"#这是一个带有特殊字符的测试字符串：!@#$%^&*(
        )
        == 16
    )

    # Test case 4: String with non-ASCII characters
    assert num_tokens_in_string("Héllò, wörld! 你好，世界！") == 17#断言字符串中的非ASCII字符的单词数为17

    # Test case 5: Long string
    long_string = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10#长字符串
    )
    assert num_tokens_in_string(long_string) == 101#断言长字符串的单词数为101

    # Test case 6: Different encoding
    assert (
        num_tokens_in_string("Hello, world!", encoding_name="p50k_base") == 4#
    )
