import unittest
from tools.publish_training_progress import public_report,should_publish

class ProgressReportingTests(unittest.TestCase):
    def test_public_report_excludes_machine_details(self):
        report=public_report(dict(stage='training',updates=2,examples_seen=8,updated_at=1,
            pid=123,checkpoint='private computer path',error='private exception'))
        self.assertNotIn('pid',report)
        self.assertNotIn('checkpoint',report)
        self.assertNotIn('error',report)
        self.assertEqual(report['examples_seen'],8)

    def test_same_day_progress_does_not_create_spam_commits(self):
        previous=dict(stage='training',phase=0,updates=1,examples_seen=4,updated_at=1,recorded_date='2026-10-10')
        current=dict(previous,updates=2,examples_seen=8,updated_at=2)
        self.assertFalse(should_publish(previous,current))
        self.assertTrue(should_publish(previous,dict(current,recorded_date='2026-10-11')))
        self.assertTrue(should_publish(previous,dict(current,examples_seen=10000)))
        self.assertTrue(should_publish(previous,dict(current,stage='failed')))
        self.assertFalse(should_publish(previous,dict(previous,recorded_date='2026-10-11')))
