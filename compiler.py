mnemonics = {
    "NOP": b'\x00\x00',
    "STA": b'\x10\x00',
    "LDA": b'\x20\x00',
    "ADD": b'\x30\x00',
    "OR":  b'\x40\x00',
    "AND": b'\x50\x00',
    "NOT": b'\x60\x00',
    "JMP": b'\x80\x00',
    "JN":  b'\x90\x00',
    "JZ":  b'\xA0\x00',
    "HLT": b'\xF0\x00'
}

import collections
import re
import types

class Lookahead:
    def __init__(self, iter):
        self.iter = iter
        self.buffer = []

    def __iter__(self):
        return self

    def next(self):
        if self.buffer:
            return self.buffer.pop(0)
        else:
            return next(self.iter)

    def lookahead(self, n = 1):
        """Return an item n entries ahead in the iteration."""
        while n > len(self.buffer):
            try:
                self.buffer.append(next(self.iter))
            except StopIteration:
                return None
        return self.buffer[n-1]

Token = collections.namedtuple('Token', ['typ', 'value', 'line', 'column'])

def tokenize(code):
    token_spec = [
        ('ID', r'[a-zA-Z_][a-zA-Z_0-9]*'),
        ('NUMBER', r'(([0-9]+)|(0x[0-9A-F]+))'),
        ('MODIFIER', r'((\.)|(:))'),
        ('NEWLINE', r'\n'),
        ('SKIP', r'([ \t]+)|(#.*\n)'),
        ('MISMATCH',r'.')
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_spec)
    line_num = 1
    line_start = 0
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group(kind)
        if kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
        elif kind == 'SKIP':
            pass
        elif kind == 'MISMATCH':
            raise RuntimeError('{} unexpected on line {}'.format(value, line_num))
        else:
            if kind == 'ID' and value in mnemonics:
                kind = 'MNEMONIC'
            column = mo.start() - line_start
            yield Token(kind, value, line_num, column)

Node = collections.namedtuple('Node', ['typ', 'nodes'])

class ParseError(Exception):

    def __init__(self, tk, expect):
        self.tk = tk
        self.expect = expect
        msg = "Unexpected {} '{}'. Expected {}. At line {} end column {}".format(tk.typ, tk.value, expect, tk.line, tk.column)
        super(ParseError, self).__init__(msg)

class Parser(object):

    def __init__(self, tokens):
        self.tokens = Lookahead(tokens);

    def parse(self):
        return self.parse_program()

    def parse_program(self):
        tks = self.tokens
        intructions = []
        while tks.lookahead():
            tk = tks.lookahead()
            if tk.typ == 'ID':
                intructions.append(self.parse_assignment())
            elif tk.typ == 'MODIFIER':
                intructions.append(self.parse_label())
            elif tk.typ == 'MNEMONIC':
                intructions.append(self.parse_mnemonic())
            else:
                raise ParseError(tk, "ID, MODIFIER or MNEMONIC")
        return Node('PROGRAM', intructions)

    def parse_assignment(self):
        var = self.tokens.next()
        value = self.tokens.next()
        if value.typ not in ["NUMBER", "ID"]:
            raise ParseError(value, "NUMBER or ID")
        return Node('ASSIGMENT', [var, value])

    def parse_mnemonic(self):
        mne = self.tokens.next()
        if mne.typ != "MNEMONIC":
            raise ParseError(mne, "MNEMONIC")
        if mne.value in ['NOP', 'NOT', 'HLT']:
            return Node('INSTRUCTION', [mne])
        else:
            addr = self.parse_address()
            return Node('INSTRUCTION', [mne, addr])

    def parse_label(self):
        tk = self.tokens.next()
        if tk.typ == 'MODIFIER' and tk.value == ':':
            ident = self.tokens.next()
            if ident.typ == 'ID':
                return Node('LABEL', [ident])
            else:
                raise ParseError(tk, "'ID'")
        else:
            raise ParseError(tk, "':'")


    def parse_address(self):
        tk = self.tokens.next()
        if tk.typ == 'ID':
            return Node('ADDRESS', [tk])
        elif tk.typ == 'MODIFIER' and tk.value == '.':
            value = self.tokens.next()
            if value.typ == 'NUMBER':
                return Node('ADDRESS', [Node('VALUE', [value])])
            else:
                raise ParseError(tk, "NUMBER")
        else:
            raise ParseError(tk, "ID or '.'")

class NodeVisitor:
    stack = []
    def genvisit(self, node):
        name = "visit_"+node.typ
        result = getattr(self, name)(node)
        if isinstance(result, types.GeneratorType):
            result = yield from result
        return result

    def visit(self, node):
        stack = [self.genvisit(node)]
        result = None
        while stack:
            try:
                node = stack[-1].send(result)
                stack.append(self.genvisit(node))
                result = None
            except Exception as e:
                stack.pop()
                result = e.value
        return result

class Prettifier(NodeVisitor):

    def indent(self, text, amount, ch=' '):
        padding = amount * ch
        return padding + ('\n'+padding).join(text.split('\n'))

    def visit_PROGRAM(self, node):
        result = "PROGRAM\n"
        for stm in node.nodes:
            stm = yield stm
            result += self.indent(stm, 3)+"\n"
        return result

    def visit_ASSIGMENT(self, node):
        result = "ASSIGMENT\n"
        for stm in node.nodes:
            stm = yield stm
            result += self.indent(stm, 3)+"\n"
        return result

    def visit_INSTRUCTION(self, node):
        result = "INSTRUCTION\n"
        for stm in node.nodes:
            stm = yield stm
            result += self.indent(stm, 3)+"\n"
        return result

    def visit_LABEL(self, node):
        result = "LABEL\n"
        for stm in node.nodes:
            stm = yield stm
            result += self.indent(stm, 3)+"\n"
        return result

    def visit_ADDRESS(self, node):
        result = "ADDRESS\n"
        for stm in node.nodes:
            stm = yield stm
            result += self.indent(stm, 3)+"\n"
        return result

    def visit_VALUE(self, node):
        result = "VALUE\n"
        for stm in node.nodes:
            stm = yield stm
            result += self.indent(stm, 3)+"\n"
        return result

    def visit_ID(self, node):
        return str(node)

    def visit_NUMBER(self, node):
        return str(node)

    def visit_MNEMONIC(self, node):
        return str(node)

class Compiler(NodeVisitor):

    def __init__(self, code_offset=0, data_offset=128):
        self.code_offset = code_offset
        self.data_offset = data_offset

        self.lookup = {}
        self.code_line = code_offset
        self.data_line = data_offset

    def visit_PROGRAM(self, node):
        magic_number = b'\x03NDR'
        instructions = []
        data = []
        for stm in node.nodes:
            ins, dat = yield stm
            instructions+=ins
            data+=dat
        instructions = map(
            lambda ins: ins(self.lookup) if isinstance(ins, collections.Callable) else ins,
            instructions)

        for k, l in self.lookup.items():
            print("{}\t\t{}".format(k, l[0]))
        instr = bytes(self.code_offset*2)+b''.join(instructions)
        data = bytes(self.data_offset*2-len(instr))+b''.join(data)
        return magic_number+instr+data

    def visit_ASSIGMENT(self, node):
        ident = node.nodes[0].value
        value = node.nodes[1]
        if value.typ == 'NUMBER':
            self.lookup[ident] = bytes([self.data_line, 0])
            self.data_line+=1
            return [], [bytes([int(value.value, base=0), 0])]
        elif value.value in self.lookup:
            self.lookup[ident] = self.lookup[value.value]
            return [], []
        else:
            raise RuntimeError("Impossible reference '{}'. Variable not defined yet.".format(value.value))


    def visit_INSTRUCTION(self, node):
        mne = node.nodes[0].value
        self.code_line+=1
        if len(node.nodes)>1:
            c, d = yield node.nodes[1]
            self.code_line+=1
            return [mnemonics[mne]]+c, d
        return [mnemonics[mne]], []

    def visit_LABEL(self, node):
        ident = node.nodes[0].value
        self.lookup[ident] = bytes([self.code_line, 0])
        return [], []

    def visit_ADDRESS(self, node):
        addr = node.nodes[0]
        if addr.typ == 'ID':
            return [lambda lt: lt[addr.value]], []
        else:
            return (yield addr)

    def visit_VALUE(self, node):
        val = node.nodes[0].value
        line = self.data_line
        self.data_line+=1
        return [bytes([line, 0])], [bytes([int(val, base=0), 0])]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('input_file', type=argparse.FileType('r'))
    p.add_argument('output_file', type=argparse.FileType('wb'))
    p.add_argument('--code_offset', type=int, default=0)
    p.add_argument('--data_offset', type=int, default=128)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    with args.input_file as input_file:
        code = input_file.read()
        tks = tokenize(code)
        ast = Parser(tks).parse()
        if args.debug:
            print(Prettifier().visit(ast))
        with args.output_file as output_file:
            output_file.write(Compiler(args.code_offset, args.data_offset).visit(ast))
