import numpy as np


class PlyObject:
    def __init__(self,header) -> None:
        self.header = header
        self.elements={k: None for k in header['element'].keys()}

    def __str__(self):
        return self.header

    def __getitem__(self, key):
        return self.elements[key]
    
    def element_info(self,key):
        return self.header['element'][key]

class PlyReader:
    TypeTable = {
        'char': (1, np.byte),
        'uchar': (1, np.ubyte),
        'short': (2, np.short),
        'ushort': (2, np.ushort),
        'int': (4, np.intc),
        'uint': (4, np.uintc),
        'float': (4, np.float32),
        'double': (8, np.float64),
    }
    def __init__(self):
        pass
    
    def _headerparse(self,f):
        header = {}
        line = f.readline()
        line = line.decode('ascii').strip().split()
        while line[0] != 'end_header':
            if line[0] == 'format':
                header['format'] = line[1]
                header['verion'] = line[2]
            elif line[0] == 'comment':
                if 'comment' in header.keys():
                    header['comment'].append(line[1:])
                else:
                    header['comment'] = [line[1:]]
            elif line[0] == 'element':
                if ('element' not in header.keys()):
                    header['element'] = {}
                elem_name = line[1]
                element ={'number': int(line[2]), 'itemsize': 0 }
                elem_type = []
                line = f.readline()
                line = line.decode('ascii').strip().split()
                while line[0]=='property':
                    if line[1] == 'list':
                        raise NotImplementedError
                    else:
                        dtype_size,dtype = self.TypeTable[line[1]]
                        elem_type.append((line[2],dtype))
                        element['itemsize']+=dtype_size
                    line = f.readline()
                    line = line.decode('ascii').strip().split()  
                element['property'] = elem_type
                header['element'][elem_name]=element
                continue                
            line = f.readline()
            line = line.decode('ascii').strip().split()
        return header

    def open(self,filename):
        with open(filename,'rb') as f:
            header = self._headerparse(f)
            plyobj = PlyObject(header)
            for elem_name, elem_info in header['element'].items():
                elem_num = elem_info['number']
                elem_itemsize = elem_info['itemsize']
                elem_dtype = elem_info['property']
                elem_bytessize = elem_num * elem_itemsize
                elem_buffer = f.read(elem_bytessize)
                plyobj.elements[elem_name]=np.frombuffer(elem_buffer,dtype=elem_dtype)
            return plyobj
            


if __name__=="__main__":
    filepath = "/home/maskjp/Code/RELLIS-3D/utils/example/frame000104-1581624663_170.ply"

    plyreader = PlyReader()
    plyobj = plyreader.open(filepath)
    print(np.max(plyobj['vertex']['intensity']))
