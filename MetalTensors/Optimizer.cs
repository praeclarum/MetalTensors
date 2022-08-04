using System;

namespace MetalTensors
{
    public abstract class Optimizer : Configurable
    {
        public const float DefaultLearningRate = 1e-3f;
        public const float DefaultGradientRescale = 1.0f;
        public const float DefaultRegularizationScale = 1.0f;

        public float LearningRate = DefaultLearningRate;
        public float GradientRescale = DefaultGradientRescale;
        public RegularizationType RegularizationType = RegularizationType.None;
        public float RegularizationScale = DefaultRegularizationScale;
    }

    public class AdamOptimizer : Optimizer
    {
        public float Beta1 { get; }
        public float Beta2 { get; }
        public float Epsilon { get; }

        public AdamOptimizer (float learningRate = DefaultLearningRate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-7f)
        {
            LearningRate = learningRate;
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;
        }
    }
}
