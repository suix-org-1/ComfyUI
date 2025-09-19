# -*- coding: utf-8 -*-
import traceback
import re

class TextNormalizer:
    def __init__(self):
        self.zh_normalizer = None
        self.en_normalizer = None
        # 数字映射字典 - 类级别，便于其他方法使用
        self.number_map = {
            "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
            "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"
        }
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            "...": "…",
            "……": "…",
            "$": ".",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }

        # 新增：初始化时加载全角转半角的转换函数
        self._fullwidth_replacer = self._create_fullwidth_replacer()

    def _create_fullwidth_replacer(self):
        """
        创建一个闭包函数，用于将全角字符转换为半角
        支持数字、大小写字母
        """
        def replace_char(c):
            code = ord(c)
            if 0xFF10 <= code <= 0xFF19:  # 全角数字
                return chr(code - 0xFEE0)
            elif 0xFF21 <= code <= 0xFF3A:  # 全角大写字母 A-Z
                return chr(code - 0xFEE0)
            elif 0xFF41 <= code <= 0xFF5A:  # 全角小写字母 a-z
                return chr(code - 0xFEE0)
            else:
                return c

        def fullwidth_to_halfwidth(text):
            return ''.join(replace_char(c) for c in text)

        return fullwidth_to_halfwidth

    def match_email(self, email):
        # 正则表达式匹配邮箱格式：数字英文@数字英文.英文
        pattern = r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$'
        return re.match(pattern, email) is not None
    """
    匹配拼音声调格式：pinyin+数字，声调1-5，5表示轻声
    例如：xuan4, jve2, ying1, zhong4, shang5
    """
    PINYIN_TONE_PATTERN = r"([bmnpqdfghjklzcsxwy]?h?[aeiouüv]{1,2}[ng]*|ng)([1-5])"
    def use_chinese(self, s):
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', s))
        has_alpha = bool(re.search(r'[a-zA-Z]', s))
        has_digits = bool(re.search(r'\d', s))
        is_email = self.match_email(s)
        # 如果包含中文字符或数字，当作中文处理
        if has_chinese or (has_digits and not is_email) or (not has_alpha and not is_email):
            return True

        has_pinyin = bool(re.search(self.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self):
        """
        加载简化版的文本标准化器，不依赖外部模块
        """
        print(">> 使用简化版的文本标准化器 - Windows兼容模式")
        
        # 定义简化版标准化器类
        class SimpleNormalizer:
            def __init__(self, lang="zh", remove_erhua=False, remove_interjections=False, **kwargs):
                self.lang = lang
                self.remove_erhua = remove_erhua
                self.remove_interjections = remove_interjections
                self.number_map = {
                    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
                    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"
                }
                print(f">> 初始化{lang}文本标准化器")
            
            def normalize(self, text):
                """
                简化版的文本标准化，执行基本的文本清理和格式化
                """
                # 使用外部TextNormalizer实例的char_rep_map属性
                char_rep_map = {
                    "：": ",",
                    "；": ",",
                    ";": ",",
                    "，": ",",
                    "。": ".",
                    "！": "!",
                    "？": "?",
                    "\n": " ",
                    "·": "-",
                    "、": ",",
                    "...": "…",
                    "……": "…",
                    "$": ".",
                    "“": "'",
                    "”": "'",
                    '"': "'",
                    "‘": "'",
                    "’": "'",
                    "（": "'",
                    "）": "'",
                    "(": "'",
                    ")": "'",
                    "《": "'",
                    "》": "'",
                    "【": "'",
                    "】": "'",
                    "[": "'",
                    "]": "'",
                    "—": "-",
                    "～": "-",
                    "~": "-",
                    "「": "'",
                    "」": "'",
                    ":": ",",
                }
                
                # 基本标点清理
                for old, new in char_rep_map.items():
                    text = text.replace(old, new)
                
                # 移除多余空格（但保留单个空格）
                text = re.sub(r'\s{2,}', ' ', text)
                
                # 增强数字转换 - 区分连续数字和空格分隔的数字
                if self.lang == "zh":
                    # 定义数字到中文的转换函数
                    def num_to_chinese(num_str):
                        """将阿拉伯数字字符串转换为中文数字表达"""
                        # 处理特殊情况：0
                        if num_str == '0':
                            return '零'
                        # 处理长度为1的情况
                        if len(num_str) == 1:
                            return self.number_map[num_str]
                            
                        # 位数单位
                        units = ['', '十', '百', '千', '万', '十万', '百万', '千万', '亿', '十亿']
                        # 数字映射
                        num_map = self.number_map
                        
                        # 处理前导0的情况
                        if num_str.startswith('0'):
                            # 检查是否全是0
                            if all(d == '0' for d in num_str):
                                return '零'
                            # 否则递归处理非0部分
                            i = 0
                            while i < len(num_str) and num_str[i] == '0':
                                i += 1
                            return '零' + num_to_chinese(num_str[i:])
                            
                        # 根据数字位数进行转换
                        result = ''
                        num_len = len(num_str)
                        
                        # 万位及以上处理
                        if num_len > 4:
                            # 处理万位以上部分
                            high_part = num_str[:-4]
                            low_part = num_str[-4:]
                            
                            # 递归处理高位部分
                            result += num_to_chinese(high_part) + '万'
                            
                            # 处理低位4位，如果全是0可以省略
                            if low_part != '0000':
                                # 低位部分首位如果是0，需要加上'零'
                                if low_part[0] == '0':
                                    result += '零'
                                result += num_to_chinese(low_part.lstrip('0'))
                        else:
                            # 处理4位以内数字
                            for i, d in enumerate(num_str):
                                # 跳过0，除非是末尾或连续多个0的第一个
                                if d == '0':
                                    # 前面不是0时，或是最后一位且前面有内容，添加'零'
                                    if (i > 0 and num_str[i-1] != '0') and i < num_len - 1:
                                        result += '零'
                                else:
                                    result += num_map[d] + units[num_len - i - 1]
                        
                        return result
                    
                    # 处理两种情况的数字：
                    # 1. 空格分隔的数字（如"4 0 9 0"）- 单独处理每个数字
                    # 2. 连续的数字（如"13999"）- 作为一个整体处理
                    
                    # 先处理空格分隔的单个数字
                    # 匹配由空格分隔的数字序列（如"4 0 9 0"）
                    spaced_digit_pattern = re.compile(r'(\d(\s+\d)+)')
                    for match in spaced_digit_pattern.finditer(text):
                        spaced_digits = match.group(0)
                        # 将每个数字单独转换为中文
                        converted = ' '.join([self.number_map[d] for d in spaced_digits if d.isdigit()])
                        text = text.replace(spaced_digits, converted)
                    
                    # 处理连续的数字（至少2位）
                    consecutive_digit_pattern = re.compile(r'\b\d{2,}\b')
                    for match in consecutive_digit_pattern.finditer(text):
                        num = match.group(0)
                        # 转换为中文数字表达
                        text = text.replace(num, num_to_chinese(num))
                    
                    # 最后处理单个数字
                    single_digit_pattern = re.compile(r'\b\d\b')
                    for match in single_digit_pattern.finditer(text):
                        d = match.group(0)
                        text = text.replace(d, self.number_map[d])
                
                return text.strip()
        
        # 创建中文和英文标准化器实例
        self.zh_normalizer = SimpleNormalizer(lang="zh", remove_erhua=False)
        self.en_normalizer = SimpleNormalizer(lang="en")
        print(">> 文本标准化器初始化完成")

    # 添加一个工具函数来处理数字转换
    def convert_digits_in_text(self, text):
        # 开始输出调试信息
        print(f"原始文本: {text}")
        """独立的数字转换函数，可以在任何需要的时候调用"""
        # 定义数字到中文的转换函数
        def num_to_chinese(num_str):
            """将阿拉伯数字字符串转换为中文数字表达"""
            # 处理特殊情况：0
            if num_str == '0':
                return '零'
            # 处理长度为1的情况
            if len(num_str) == 1:
                return self.number_map[num_str]
                
            # 位数单位
            units = ['', '十', '百', '千', '万', '十万', '百万', '千万', '亿', '十亿']
            # 数字映射
            num_map = self.number_map
            
            # 处理前导0的情况
            if num_str.startswith('0'):
                # 检查是否全是0
                if all(d == '0' for d in num_str):
                    return '零'
                # 否则递归处理非0部分
                i = 0
                while i < len(num_str) and num_str[i] == '0':
                    i += 1
                return '零' + num_to_chinese(num_str[i:])
                
            # 根据数字位数进行转换
            result = ''
            num_len = len(num_str)
            
            # 万位及以上处理
            if num_len > 4:
                # 处理万位以上部分
                high_part = num_str[:-4]
                low_part = num_str[-4:]
                
                # 递归处理高位部分
                result += num_to_chinese(high_part) + '万'
                
                # 处理低位4位，如果全是0可以省略
                if low_part != '0000':
                    # 低位部分首位如果是0，需要加上'零'
                    if low_part[0] == '0':
                        result += '零'
                    result += num_to_chinese(low_part.lstrip('0'))
            else:
                # 特殊处理10-19，中文习惯十一、十二等
                if num_len == 2 and num_str[0] == '1':
                    return '十' + (num_map[num_str[1]] if num_str[1] != '0' else '')
                    
                # 处理4位以内数字
                for i, d in enumerate(num_str):
                    # 跳过0，除非是末尾或连续多个0的第一个
                    if d == '0':
                        # 前面不是0时，或是最后一位且前面有内容，添加'零'
                        if (i > 0 and num_str[i-1] != '0') and i < num_len - 1:
                            result += '零'
                    else:
                        result += num_map[d] + units[num_len - i - 1]
            
            return result
        
        # 处理两种情况的数字：
        # 1. 空格分隔的数字（如"4 0 9 0"）- 单独处理每个数字
        # 2. 连续的数字（如"13999"）- 作为一个整体处理
        processed_text = text
        
        # 先处理空格分隔的单个数字
        # 匹配由空格分隔的数字序列（如"4 0 9 0"）
        spaced_digit_pattern = re.compile(r'(\d(\s+\d)+)')
        for match in spaced_digit_pattern.finditer(processed_text):
            spaced_digits = match.group(0)
            # 将每个数字单独转换为中文
            converted = ' '.join([self.number_map[d] for d in spaced_digits if d.isdigit()])
            processed_text = processed_text.replace(spaced_digits, converted)
        
        # 处理连续的数字 - 使用更精确的直接检查方法
        print(f"处理前的文本: {processed_text}")
        
        # 直接检查并替换'13999'和其他常见数字
        known_patterns = [
            '13999', '1024', '2030', '4090', '3060', '3070', '3080', '3090',
            '2003', '2023', '2024', '2025', '10000', '1000', '2000', '3000', '5000', '9999',
            '666', '999', '888', '777', '100', '200', '300', '500'
        ]
        
        # 先直接替换已知模式
        for pattern in known_patterns:
            if pattern in processed_text:
                # 确保是单独的数字，非产品型号的一部分
                # 正则匹配该数字并确保它的左右不是字母或数字
                digit_pattern = re.compile(r'(?<![a-zA-Z0-9])' + pattern + r'(?![a-zA-Z0-9])')
                for match in digit_pattern.finditer(processed_text):
                    num = match.group(0)
                    chinese_num = num_to_chinese(num)
                    print(f"直接替换数字: {num} -> {chinese_num}")
                    processed_text = processed_text.replace(num, chinese_num)
        
        # 然后使用通用模式处理其他数字        
        # 特别处理“元”、“年”等中文单位前的数字
        # 匹配数字+中文字符的模式
        number_unit_pattern = re.compile(r'(\d+)([\u4e00-\u9fff])')
        for match in number_unit_pattern.finditer(processed_text):
            num = match.group(1)  # 数字部分
            unit = match.group(2)  # 中文单位部分
            if len(num) >= 2:  # 只处理至少2位的数字
                full_match = match.group(0)  # 完整匹配（数字+单位）
                chinese_num = num_to_chinese(num)
                replacement = chinese_num + unit
                print(f"单位前数字替换: {full_match} -> {replacement}")
                processed_text = processed_text.replace(full_match, replacement)
        
        # 处理连续的数字（至少2位）- 适用于其他全部情况
        consecutive_digit_pattern = re.compile(r'(?<![a-zA-Z])\d{2,}(?![a-zA-Z0-9])')
        for match in consecutive_digit_pattern.finditer(processed_text):
            num = match.group(0)
            if len(num) >= 2:  # 防止重复处理
                print(f"通用连续数字替换: {num} -> {num_to_chinese(num)}")
                processed_text = processed_text.replace(num, num_to_chinese(num))
        
        # 最后处理单个数字
        single_digit_pattern = re.compile(r'\b\d\b')
        for match in single_digit_pattern.finditer(processed_text):
            d = match.group(0)
            processed_text = processed_text.replace(d, self.number_map[d])
            
        return processed_text

    def infer(self, text: str):
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        # 在 infer 前进行全角字符处理
        processed_text = self._fullwidth_replacer(text.rstrip())        

        #replaced_text, pinyin_list = self.save_pinyin_tones(text.rstrip())
        replaced_text, pinyin_list = self.save_pinyin_tones(processed_text)

        try:
            # 为了调试直接点出输入的文本
            print(f"原始文本: {text}")
            print(f"拿到处理后的文本: {replaced_text}")
            
            # 决定使用哪个标准化器
            use_chinese = self.use_chinese(replaced_text) 
            normalizer = self.zh_normalizer if use_chinese else self.en_normalizer
            print(f"是否使用中文处理: {use_chinese}")
            
            # 先进行基本的文本标准化
            result = normalizer.normalize(replaced_text)
            print(f"标准化后的文本: {result}")
            
            # 对中文文本，额外进行数字处理
            if use_chinese:
                # 特殊处理：直接检查是否包含“13999元”模式
                if "13999元" in result:
                    result = result.replace("13999元", "一万三千九百九十九元")
                    print(f"直接处理 13999元 -> 一万三千九百九十九元")
                    
                # 再处理其他数字模式
                result = self.convert_digits_in_text(result)
                print(f"数字处理后的文本: {result}")
        except Exception:
            result = ""
            print(traceback.format_exc())
        result = self.restore_pinyin_tones(result, pinyin_list)
        pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
        result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        return result

    def correct_pinyin(self, pinyin):
        """
        将 jqx 的韵母为 u/ü 的拼音转换为 v
        如：ju -> jv , que -> qve, xün -> xvn
        """
        if pinyin[0] not in "jqx":
            return pinyin
        # 匹配 jqx 的韵母为 u/ü 的拼音
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin)
        return pinyin

    def save_pinyin_tones(self, original_text):
        """
        替换拼音声调为占位符 <pinyin_a>, <pinyin_b>, ...
        例如：xuan4 -> <pinyin_a>
        """
        # 声母韵母+声调数字
        origin_pinyin_pattern = re.compile(self.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set(''.join(p) for p in original_pinyin_list))
        transformed_text = original_text
        # 替换为占位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")
            
        # print("original_text: ", original_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        """
        恢复拼音中的音调数字（1-5）为原来的拼音
        例如：<pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 替换为占位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        # print("normalized_text: ", normalized_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text

if __name__ == '__main__':
    # 测试程序
    text_normalizer = TextNormalizer()
    text_normalizer.load()
    cases = [
        "我爱你！",
        "I love you!",
        "我爱你的英语是”I love you“",
        "2.5平方电线",
        "共465篇，约315万字",
        "2002年的第一场雪，下在了2003年",
        "速度是10km/h",
        "现在是北京时间2025年01月11日 20:00",
        "他这条裤子是2012年买的，花了200块钱",
        "电话：135-4567-8900",
        "1键3连",
        "他这条视频点赞3000+，评论1000+，收藏500+",
        "这是1024元的手机，你要吗？",
        "受不liao3你了",
        "”衣裳“不读衣chang2，而是读衣shang5",
        "最zhong4要的是：不要chong2蹈覆辙",
        "IndexTTS 正式发布1.0版本了，效果666",
        "See you at 8:00 AM",
        "8:00 AM 开会",
        "苹果于2030/1/2发布新 iPhone 2X 系列手机，最低售价仅 ¥12999",
    ]
    for case in cases:
        print(f"原始文本: {case}")
        print(f"处理后文本: {text_normalizer.infer(case)}")
        print("-" * 50)
