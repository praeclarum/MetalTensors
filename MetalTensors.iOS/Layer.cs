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

        readonly ConcurrentDictionary<string, MPSNNFilterNode> deviceFilterNodes =
            new ConcurrentDictionary<string, MPSNNFilterNode> ();

        public abstract int InputCount { get; }

        public string Label => label;

        protected Layer (string? label = null)
        {
            var id = Interlocked.Increment (ref nextId);
            this.label = string.IsNullOrWhiteSpace (label) ? GetType ().Name + id : label!;
        }

        public override string ToString () => Label;

        public abstract int[] GetOutputShape (params Tensor[] inputs);

        public MPSNNImageNode GetMetalImageNode (Tensor[] inputs, bool training, IMTLDevice device)
        {
            var f = GetFilterNode (inputs, training, device);
            //Console.WriteLine (f.ResultImage.DebugDescription);
            return f.ResultImage;
        }

        public Tensor GetOutput (params Tensor[] inputs)
        {
            return new LayerTensor (this, inputs);
        }

        public Task<Tensor> ExecuteAsync (Tensor[] inputs, IMTLDevice device)
        {
            if (inputs.Length < InputCount)
                throw new ArgumentException (nameof (inputs));

            var tcs = new TaskCompletionSource<Tensor> ();
            ThreadPool.QueueUserWorkItem (StartGraph);
            return tcs.Task;

            void StartGraph (object s)
            {
                try {
                    var node = GetFilterNode (inputs, false, device);

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

        MPSNNFilterNode GetFilterNode (Tensor[] inputs, bool training, IMTLDevice device)
        {
            var inputImageNodes = inputs.Select (x => (x.GetMetalImageNode (training, device), x.Shape)).ToArray ();
            return GetFilterNode (inputImageNodes, training, device);
        }

        MPSNNFilterNode GetFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, bool training, IMTLDevice device)
        {
            var key = device.Handle + "-" + string.Join (",", inputs.Select (x => x.ImageNode.MPSHandle.Label));
            if (deviceFilterNodes.TryGetValue (key, out var node))
                return node;

            node = CreateFilterNode (inputs, device);
            node.ResultImage.MPSHandle = new LayerHandle (this);
            node.Label = Label;

            deviceFilterNodes.TryAdd (key, node);
            return node;
        }

        protected abstract MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device);
    }
}
