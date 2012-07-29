from util import *

class Key(object):

    def __init__(self, root):
        self.root = root

    def __hash__(self):
        return self.root

    def __eq__(self, other):
        return type(self) == type(other) and hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def compare(self, other):
        if self == other:
            return KeySame()

        if type(self) == type(other) and \
                ((self.root - other.root) % 12 == 7 or \
                     (other.root - self.root) % 12 == 7):
            return KeyFifth()

        if type(self) != type(other) and \
                isinstance(other, Key) and \
                self.root == other.root:
            return KeyParallel()

        return KeyOther()

class MajorKey(Key):

    def __repr__(self):
        return '<MajorKey: %s>' % note_names[self.root]

    def compare(self, other):
        if isinstance(other, MinorKey) and \
                (self.root - 3) % 12 == other.root:
            return KeyRelative()
        return Key.compare(self, other)
    
class MinorKey(Key):

    def __repr__(self):
        return '<MinorKey: %s>' % note_names[self.root]

    def compare(self, other):
        if isinstance(other, MajorKey) and \
                (self.root + 3) % 12 == other.root:
            return KeyRelative()
        return Key.compare(self, other)


# Based on http://www.music-ir.org/mirex/wiki/2012:Audio_Key_Detection#Evaluation_Procedures

class KeyDiff(object):
    def __hash__(self):
        return 0
    def __eq__(self, other):
        return type(self) == type(other) and hash(self) == hash(other)

class KeySame(KeyDiff):
    def score(self):
        return 1.0
    def name(self):
        return 'Same'

class KeyFifth(KeyDiff):
    def score(self):
        return 0.5
    def name(self):
        return 'Perfect Fifth'

class KeyRelative(KeyDiff):
    def score(self):
        return 0.3
    def name(self):
        return 'Relative Major/Minor'

class KeyParallel(KeyDiff):
    def score(self):
        return 0.2
    def name(self):
        return 'Parallel Major/Minor'

class KeyOther(KeyDiff):
    def score(self):
        return 0.0
    def name(self):
        return 'Other'
