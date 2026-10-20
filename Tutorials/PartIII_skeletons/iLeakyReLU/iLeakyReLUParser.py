# ----------------------------------------------------------------------
# File: iLeakyReLUParser.py  (SoCDAML Part III - Step 2 skeleton)
#
# Paste this class into:
#   Deeploy/Targets/Generic/Parsers.py
#
# Imports already present in that file (math, numpy as np,
# onnx_graphsurgeon as gs, NodeParser, NetworkContext).
# ----------------------------------------------------------------------
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


class iLeakyReLUParser(NodeParser):

    def __init__(self):
        super().__init__()

    def parseNode(self, node: gs.Node) -> bool:
        # one input, exactly one output, and both 'mul' and 'shift'
        # attributes. On success, store them into
        # self.operatorRepresentation as ints and return True.
        
        # one input, exactly one output
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            return False
        # check if there are mul and shift
        ret = all(['mul' in node.attrs, 'shift' in node.attrs])

        # store them as int
        if ret:
            mul = int(node.attrs['mul'])
            shift = int(node.attrs['shift'])
            
            if mul != 1 or not 0 <= shift < 8:
                return False
                    
            self.operatorRepresentation['mul'] = mul
            self.operatorRepresentation['shift'] = shift

        return ret

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node, channels_first: bool = True):
        # using node.inputs[0].name / node.outputs[0].name, and populate
        # self.operatorRepresentation with:
        #     'data_in'  -> input tensor name
        #     'data_out' -> output tensor name
        #     'size'     -> int(np.prod(input_shape))
        # Return (ctxt, True).
        
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['size'] = int(np.prod(data_in.shape))
        
        return ctxt, True
