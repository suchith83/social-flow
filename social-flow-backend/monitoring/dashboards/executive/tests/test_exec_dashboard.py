# Unit tests for executive dashboard
"""
Unit Tests for Executive Dashboard
"""

import unittest
from monitoring.dashboards.executive.kpi_adapter import KPIAdapter
from monitoring.dashboards.executive.trend_analysis import TrendAnalyzer
from monitoring.dashboards.executive.sla_overlay import SLAOverlay


class TestExecDashboard(unittest.TestCase):
    def setUp(self):
        self.adapter = KPIAdapter()
        self.trend = TrendAnalyzer()
        self.sla = SLAOverlay()

    def test_fetch_kpi(self):
        data = self.adapter.fetch_kpi("business_revenue_daily", 5)
        self.assertEqual(len(data), 5)
        self.assertTrue(all(isinstance(v, (float, int)) for v in data))

    def test_trend_analysis(self):
        values = [100, 120, 150, 180]
        trend = self.trend.analyze("revenue", values)
        self.assertEqual(trend, "Positive trend (growth)")

    def test_sla_overlay(self):
        values = [99.0, 98.5]
        thresholds = {"warning": 99.5, "critical": 98.0}
        status = self.sla.evaluate("uptime", values, thresholds)
        self.assertIn("WARNING", status)


if __name__ == "__main__":
    unittest.main()
