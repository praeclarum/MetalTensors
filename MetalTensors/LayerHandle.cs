using Foundation;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public class LayerHandle : NSObject, IMPSHandle
    {
        public string Label { get; }
        public Layer Layer { get; }

        public LayerHandle (Layer layer)
        {
            Label = layer.Name;
            Layer = layer;
        }

        public override string ToString () => Label;

        public void EncodeTo (NSCoder encoder)
        {
            encoder.Encode (new NSString (Label), "label");
        }
    }
}
