using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Layers;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public abstract class Layer
    {
        static int nextId = 1;
        readonly string label;

        readonly ConcurrentDictionary<string, MPSNNFilterNode> cachedFilterNodes =
            new ConcurrentDictionary<string, MPSNNFilterNode> ();

        public abstract int MinInputCount { get; }

        public string Label => label;

        protected Layer (string? label = null)
        {
            var id = Interlocked.Increment (ref nextId);
            this.label = string.IsNullOrWhiteSpace (label) ? GetType ().Name + id : label!;
        }

        public override string ToString () => Label;

        public virtual void ValidateInputShapes (params Tensor[] inputs)
        {
            // All input shapes are OK
        }

        public abstract int[] GetOutputShape (params Tensor[] inputs);

        public MPSNNImageNode GetMetalImageNode (Tensor[] inputs, MetalImageNodeContext context)
        {
            var f = GetFilterNode (inputs, context);
            //Console.WriteLine (f.ResultImage.DebugDescription);
            return f.ResultImage;
        }

        public Tensor GetOutput (params Tensor[] inputs)
        {
            return new LayerTensor (this, inputs);
        }

        public Task<Tensor> ExecuteAsync (Tensor[] inputs, IMTLDevice device)
        {
            if (inputs.Length < MinInputCount)
                throw new ArgumentException (nameof (inputs));

            var tcs = new TaskCompletionSource<Tensor> ();
            ThreadPool.QueueUserWorkItem (StartGraph);
            return tcs.Task;

            void StartGraph (object s)
            {
                try {
                    var context = new MetalImageNodeContext (label + " Execute", false, device);
                    var node = GetFilterNode (inputs, context);

                    using var graph = new MPSNNGraph (device, node.ResultImage, true) {
                        Format = MPSImageFeatureChannelFormat.Float32,
                    };
                    //Console.WriteLine (graph.DebugDescription);

                    var sourceHandles = graph.SourceImageHandles;
                    var sources = sourceHandles.Select (x => ((TensorHandle)x).Tensor.GetMetalImage (device)).ToArray ();

                    var r = graph.Execute (sources, (image, error) => {
                        if (error != null) {
                            tcs.SetException (new Exception (error.Description));
                        }
                        else {
                            var t = new MPSImageTensor (image);
                            tcs.SetResult (t);
                        }
                    });
                }
                catch (Exception ex) {
                    tcs.TrySetException (ex);
                }
            }
        }

        public virtual MPSCnnConvolutionDataSource? GetMetalConvDataSource (IMTLDevice device)
        {
            return null;
        }

        static IMTLDevice? FindDevice (Tensor[] tensors)
        {
            // TODO: Scan inputs for the correct device to use
            return null;
        }

        MPSNNFilterNode GetFilterNode (Tensor[] inputs, MetalImageNodeContext context)
        {
            var inputImageNodes = inputs.Select (x => (x.GetMetalImageNode (context), x.Shape)).ToArray ();

            var key = context.CacheKey + " + (" + string.Join (",", inputImageNodes.Select (x => x.Item1.MPSHandle.Label)) + ")";
            if (cachedFilterNodes.TryGetValue (key, out var node))
                return node;

            node = CreateFilterNode (inputImageNodes, context);

            if (cachedFilterNodes.TryAdd (key, node))
                return node;
            return cachedFilterNodes[key];
        }

        MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, MetalImageNodeContext context)
        {
            var node = CreateFilterNode (inputs, context.Device);

            node.ResultImage.MPSHandle = new LayerHandle (this);
            node.Label = Label;

            return node;
        }

        protected abstract MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device);
    }
}
