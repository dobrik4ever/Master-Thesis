import numpy as np

class Chunkizer:
    """Class for chunks generation. Required to fit in model a huge stack
    chunk by chunk.
    """

    def __init__(self, input_shape:tuple, chunk_shape:tuple, batch_size = 1):
        """Is used to disassemble and assemble back input stacks

        Args:
            input_shape (tuple): shape of stack, that must be divided
            chunk_shape (tuple): shape of chunk, that must be generated.
            batch_size (int): number of chunks to be fed in network
            Note, input_shape / chunk_shape, must return int values. Otherwise
            stack is truncated.
        """
        self.input_shape = np.array(input_shape)
        self.chunk_shape = np.array(chunk_shape)
        self.batch_size = batch_size
        self._check_shapes()

    def _check_shapes(self):
        condition_1 = np.any(self.input_shape % self.chunk_shape != 0)
        if condition_1:
            raise ValueError(f'Chunk shape, does not give round chunk number, {self.input_shape = }, {self.chunk_shape = }')
    
    def _check_input_shape(self, stack):
        if stack.shape != tuple(self.input_shape):
            text = f'Stack shape != input_shape! {stack.shape = }, {self.input_shape = }'
            raise ArithmeticError(text)

    def divide(self, stack):
        """Method divide. Creates a map of chunks of shape [Y, X, Z, y, x, z]

        Args:
            stack (_type_): _description_
        """
        if len(stack.shape) == 4:
            maps = []
            for i in range(stack.shape[0]):
                maps.append([self.divide(stack[i])])
            return np.vstack(maps)

        self._check_input_shape(stack)
        MAP = np.zeros([*self.indices, *self.chunk_shape])
        I, J, K = self.chunk_shape
        for i in range(self.indices[0]):
            for j in range(self.indices[1]):
                for k in range(self.indices[2]):
                    chunk = stack[
                        i * I : (i + 1) * I,
                        j * J : (j + 1) * J,
                        k * K : (k + 1) * K
                        ]
                    MAP[i,j,k] = chunk

        self.MAP = MAP
        return MAP

    @property
    def indices(self):
        return self.input_shape // self.chunk_shape
    
    def assemble(self, MAP):
        I, J, K = self.chunk_shape
        arr = np.zeros(self.input_shape)
        for i in range(self.indices[0]):
            for j in range(self.indices[1]):
                for k in range(self.indices[2]):
                    arr[
                        i * I : (i + 1) * I,
                        j * J : (j + 1) * J,
                        k * K : (k + 1) * K
                        ] = MAP[i,j,k]

        self.arr = arr
        return arr