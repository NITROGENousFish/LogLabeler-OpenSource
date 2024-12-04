import ast
import inspect


def count_return_values_in_function(func):
    # 获取函数体的 AST 节点
    source = inspect.getsource(func)
    tree = ast.parse(source)

    # 访问函数体的 AST 节点
    class ReturnVisitor(ast.NodeVisitor):
        def __init__(self):
            self.return_count = 0

        def visit_Return(self, node):
            # 每个 return 语句都被访问到
            if isinstance(node.value, ast.Tuple):
                self.return_count += len(node.value.elts)
            else:
                self.return_count += 1
            self.generic_visit(node)

    visitor = ReturnVisitor()
    visitor.visit(tree)
    return visitor.return_count


# 示例函数
def function_with_multiple_returns(a, b):
    return a, b


def function_with_single_return(a):
    return a


def function_with_no_return():
    pass


def function_with_conditional_return(x):
    if x > 0:
        return x
    else:
        return x, x * 2


def TTT(x):
    return (1,2,43,4,5,5,123,312)


# 计算返回值数量
print(count_return_values_in_function(TTT))
print(count_return_values_in_function(function_with_multiple_returns))  # 输出: 2
print(count_return_values_in_function(function_with_single_return))  # 输出: 1
print(count_return_values_in_function(function_with_no_return))  # 输出: 0
print(
    count_return_values_in_function(function_with_conditional_return)
)  # 输出: 2 (根据条件，可能有多个返回值)
