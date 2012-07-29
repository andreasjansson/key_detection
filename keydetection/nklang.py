import math

class Nklang(object):

    def get_number(self):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

class Nullklang(Nklang):

    def __init__(self):
        pass

    def get_name(self):
        return '-'

    def get_number(self):
        return -1

    def transpose(self, _):
        return Nullklang();

    def __repr__(self):
        return '<Nullklang>'

class Einklang(Nklang):

    def __init__(self, n):
        self.n = n
    
    def get_name(self):
        return note_names[self.n]

    # effectively a zweiklang, with n1 == n2
    def get_number(self):
        return self.n + 12 * self.n

    def transpose(self, delta):
        return Einklang((self.n + delta) % 12)

    def __repr__(self):
        return '<Einklang: %s>' % note_names[self.n]

class Zweiklang(Nklang):

    def __init__(self, first, second):
        if first is not None and second is not None and first > second:
            self.first = second
            self.second = first
        elif first is None and second is not None:
            raise Exception('Only second is not allowed')
        else:
            self.first = first
            self.second = second
        
    def get_name(self):
        return note_names[self.first] + ', ' + note_names[self.second]

    def get_number(self):
        return self.first + 12 * self.second
        
    def transpose(self, delta):
        return Zweiklang((self.first + delta) % 12, (self.second + delta) % 12)

    def __repr__(self):
        return '<Zweiklang: %s, %s>' % (note_names[self.first], note_names[self.second])

def klang_number_to_name(number):
    if number == -1:
        return 'Silence'
    else:
        first = number % 12
        second = int(math.floor(number / 12))
        if first == second:
            return note_names[first]
        else:
            return note_names[first] + ', ' + note_names[second]

