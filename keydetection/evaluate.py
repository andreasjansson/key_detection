from key import *

class Scoreboard(object):

    def __init__(self):
        self.diffs = {
            KeySame(): 0,
            KeyFifth(): 0,
            KeyRelative(): 0,
            KeyParallel(): 0,
            KeyOther(): 0
            }

    def add(self, diff):
        self.diffs[diff] += 1

    def get_score(self):
        score = 0
        for diff, count in self.diffs.iteritems():
            score += diff.score() * count
        return score

    def max_score(self):
        return sum(self.diffs.values())

    def print_scores(self):
        score_percent = 0 if self.max_score() == 0 \
            else self.get_score() / self.max_score() * 100

        print '\nTotal score: %.2f / %.2f (%.2f%%)\n' % (
            self.get_score(), self.max_score(), score_percent)
        for diff, count in self.diffs.iteritems():
            print '%s: %d' % (diff.name(), count)

