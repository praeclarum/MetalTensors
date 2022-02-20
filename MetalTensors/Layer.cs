using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
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
    public abstract class Layer : Configurable
    {
        readonly string name;

        readonly ConcurrentDictionary<string, MPSNNFilterNode> cachedFilterNodes =
            new ConcurrentDictionary<string, MPSNNFilterNode> ();

        public abstract int MinInputCount { get; }

        public string Name => name;

        public virtual bool IsTrainable {
            get => false;
            set { }
        }

        protected Layer (string? name = null)
        {
            this.name = string.IsNullOrWhiteSpace (name) ? GetType ().Name + Id : name!;
        }

        public override string ToString () => Name;

        public override Config Config => base.Config.Update (new Config {
            { "name", Name },
        });

        public virtual void ValidateInputShapes (params Tensor[] inputs)
        {
            // All input shapes are OK
        }

        public abstract int[] GetOutputShape (params Tensor[] inputs);

        public virtual MPSNNImageNode GetImageNode (Tensor[] inputs, MetalImageNodeContext context)
        {
            var filterNode = GetFilterNode (inputs, context);
            //Console.WriteLine (f.ResultImage.DebugDescription);
            return filterNode.ResultImage;
        }

        public virtual Tensor Call (params Tensor[] inputs)
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
                    var context = new MetalImageNodeContext (name + " Execute", false, device);
                    var node = GetImageNode (inputs, context);
                    using var graph = new MPSNNGraph (device, node, true) {
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

        public virtual MPSNNFilterNode GetFilterNode (Tensor[] inputs, MetalImageNodeContext context)
        {
            var inputImageNodes = inputs.Select (x => (x.GetImageNode (context), x.Shape)).ToArray ();

            var key = context.CacheKey + " + (" + string.Join (",", inputImageNodes.Select (x => x.Item1?.MPSHandle?.Label ?? "Unknown")) + ")";
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
            node.Label = Name;

            return node;
        }

        protected abstract MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device);
    }
}
