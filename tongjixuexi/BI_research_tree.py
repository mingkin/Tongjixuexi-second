class Node:
    def __init__(self,right=None,left=None,parent=None,value=None):
        self.right=right
        self.left=left
        self.parent=parent
        self.value=value

class Bi_tree:
    def __init__(self,data):
        self.root = None
        self.data = data

    def build_tree(self):
        # for i in self.data:
        #     self.insert_data(i)

        for i in self.data:
           self.root = self.insert_data(self.root,i)

    def insert_data(self,node,val):
        if not node:
            node = Node(value=val)
        elif val < node.value:
            node.left = self.insert_data(node.left, val)
            node.left.parent = node
        elif val > node.value:
            node.right = self.insert_data(node.right, val)
            node.right.parent = node
        return node

    def insert_no_data(self,val):
        p = self.root
        if not p:
            self.root = Node(value=val)
            return
        while True:
            if val < p.value:
                if p.left:
                    p = p.left
                else:
                    p.left = Node(parent = p,value=val)
                    return
            elif val > p.value:
                if p.right:
                    p = p.right
                else:
                    p.right = Node(parent=p, value=val)
                    return
            else:
                return

            

    def in_order(self,root):
        if root:
            self.in_order(root.left)
            print(root.value)
            self.in_order(root.right)

def main():
    tree = Bi_tree([4,6,7,9,2,1,3,5,8])
    tree.build_tree()
    tree.in_order(tree.root)

if __name__ == '__main__':
    main()