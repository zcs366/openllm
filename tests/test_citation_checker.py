"""citation_checker.py 测试。"""
import pytest
from openllm.core.citation_checker import check_citations, CitationReport, ClaimMatch


class TestCheckCitations:
    """check_citations 核心测试。"""

    def test_empty_text(self):
        """空文本返回零报告。"""
        report = check_citations("")
        assert report.total_claims == 0
        assert report.citation_rate == 0.0

    def test_whitespace_only(self):
        """纯空白返回零报告。"""
        report = check_citations("   \n\t  ")
        assert report.total_claims == 0

    def test_no_claims_no_citations(self):
        """纯叙述文本（无事实声明）返回零。"""
        report = check_citations("今天天气很好。")
        assert report.total_claims == 0

    def test_url_detected(self):
        """URL被识别为事实声明。"""
        text = "参见 https://arxiv.org/abs/2301.00001 了解详情"
        report = check_citations(text)
        assert report.total_claims >= 1
        assert any(d.pattern == "url" for d in report.details)

    def test_url_with_citation(self):
        """带标注的URL被标记为已引用。"""
        text = "参见 https://arxiv.org/abs/2301.00001 [来源: arXiv官网]"
        report = check_citations(text)
        assert report.cited_claims >= 1
        assert report.citation_rate > 0.0

    def test_paper_citation_detected(self):
        """论文引用格式被检测。"""
        text = "如 Vaswani et al., 2017 所述，注意力机制是核心。"
        report = check_citations(text)
        assert report.total_claims >= 1
        assert any(d.pattern == "paper" for d in report.details)

    def test_paper_citation_with_tag(self):
        """论文引用+来源标注。"""
        text = "如 Vaswani et al., 2017 [来源: Attention Is All You Need] 所述。"
        report = check_citations(text)
        assert report.cited_claims >= 1

    def test_date_detected(self):
        """日期格式被检测。"""
        text = "该项目于2024年3月15日启动。"
        report = check_citations(text)
        assert report.total_claims >= 1
        assert any(d.pattern == "date" for d in report.details)

    def test_percentage_detected(self):
        """百分比/统计数据被检测。"""
        text = "准确率达到95.3%。"
        report = check_citations(text)
        assert report.total_claims >= 1
        assert any(d.pattern == "statistic" for d in report.details)

    def test_number_detected(self):
        """大数字被检测。"""
        text = "全球有1,400,000名开发者使用该框架。"
        report = check_citations(text)
        assert report.total_claims >= 1
        assert any(d.pattern == "number" for d in report.details)

    def test_uncited_claims(self):
        """无标注的声明被标记为未引用。"""
        text = "OpenAI成立于2015年，有超过1000名员工。"
        report = check_citations(text)
        assert report.uncited_claims > 0

    def test_mixed_cited_and_uncited(self):
        """混合引用和未引用。"""
        text = (
            "该项目于2023年启动 [来源: 官方公告]。"
            "目前已有10万用户。"
            "如 Smith et al., 2022 所述，效果显著 [来源: 论文]。"
        )
        report = check_citations(text)
        # 至少检测到日期+数字+论文引用
        assert report.total_claims >= 3
        assert report.cited_claims >= 2
        assert report.uncited_claims >= 1

    def test_unverified_marker(self):
        """⚠️未验证标注也算有引用。"""
        text = "据称该系统支持100种语言 ⚠️未验证"
        report = check_citations(text)
        assert report.cited_claims >= 1

    def test_citation_rate_calculation(self):
        """引用率计算正确。"""
        text = (
            "事实A [来源: 来源1]"
            "事实B"
            "事实C [来源: 来源2]"
        )
        report = check_citations(text)
        assert 0.0 <= report.citation_rate <= 1.0

    def test_chinese_person_name(self):
        """中文人名被检测。"""
        text = "张三表示该技术已成熟。"
        report = check_citations(text)
        assert report.total_claims >= 1
        assert any(d.pattern == "person" for d in report.details)

    def test_english_person_name(self):
        """英文人名被检测。"""
        text = "Elon Musk announced the new product."
        report = check_citations(text)
        assert report.total_claims >= 1

    def test_arxiv_id_detected(self):
        """arXiv ID格式被检测。"""
        text = "参考 arXiv: 2301.00001 的方法论。"
        report = check_citations(text)
        assert report.total_claims >= 1
        assert any(d.pattern == "paper" for d in report.details)

    def test_partial_date_year_only(self):
        """仅年份格式也被检测。"""
        text = "该项目于2023年启动。"
        report = check_citations(text)
        assert report.total_claims >= 1
        assert any(d.pattern == "date" for d in report.details)

    def test_chinese_number_unit(self):
        """中文单位数字被检测（如"10万"）。"""
        text = "目前有10万活跃用户。"
        report = check_citations(text)
        assert report.total_claims >= 1
        assert any(d.pattern == "number" for d in report.details)


class TestCitationReport:
    """CitationReport 数据结构测试。"""

    def test_report_fields(self):
        """报告包含所有必要字段。"""
        report = CitationReport(total_claims=5, cited_claims=3, uncited_claims=2)
        assert report.citation_rate == pytest.approx(0.6)
        assert report.total_claims == 5
        assert report.cited_claims == 3
        assert report.uncited_claims == 2

    def test_report_zero_claims(self):
        """零声明报告引用率为0。"""
        report = CitationReport()
        assert report.citation_rate == 0.0

    def test_claim_match_fields(self):
        """ClaimMatch 字段完整。"""
        cm = ClaimMatch(text="test", pattern="url", has_citation=True)
        assert cm.text == "test"
        assert cm.pattern == "url"
        assert cm.has_citation is True
