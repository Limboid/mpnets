[] custom node types (not just dense, but also convolutional)
[x] custom pooling operator (sum, random, mean, max, min)
[x] add dropout
[x] add BN
[x] add parameter to disable spiking
[] support a named `reward` parameter
[] the larger network should support other node types such as
    - https://github.com/ridgerchu/SpikeGPT
    - https://github.com/BlinkDL/RWKV-LM
    - my self organizing maps library, reimplemented in pytorch. Actually just implement teh unsupervized library
    - 
[] Forewar-forward learnign
[] small circuit local feedback alignment

---

```python
MPNet(
    nodes={
        'nodeA': Node((64,64,DIMS), ...),
        'nodeB': nodeB := Node((16,16,DIMS), ...),
    },
    edges=[
        ('nodeA', 'nodeB'),
        Edge('nodeA', 'nodeB', bidirectional=True),
        SparseEdge(nodeB, 'nodeA.param1', sparsity=0.1,)
    ]
)
```