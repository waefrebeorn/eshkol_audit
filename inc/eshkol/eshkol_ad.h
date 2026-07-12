#ifndef ESHKOL_AD_H
#define ESHKOL_AD_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "eshkol_object.h"

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════
// END OBJECT HEADER SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

// Dual number helper functions for forward-mode automatic differentiation
/**
 * @brief Construct a dual number from a value and its derivative.
 * @param value Function value f(x).
 * @param derivative Derivative value f'(x).
 * @return An eshkol_dual_number_t with both fields set.
 */
static inline eshkol_dual_number_t eshkol_make_dual(double value, double derivative) {
    eshkol_dual_number_t result;
    result.value = value;
    result.derivative = derivative;
    return result;
}

/**
 * @brief Read the value component f(x) of a dual number.
 * @param d Dual number to read.
 * @return The `value` field.
 */
static inline double eshkol_dual_value(const eshkol_dual_number_t* d) {
    return d->value;
}

/**
 * @brief Read the derivative component f'(x) of a dual number.
 * @param d Dual number to read.
 * @return The `derivative` field.
 */
static inline double eshkol_dual_derivative(const eshkol_dual_number_t* d) {
    return d->derivative;
}

// ===== COMPUTATIONAL GRAPH NODE TYPES =====
// AD node types for reverse-mode automatic differentiation

/**
 * @brief Operation kind recorded at each node of the reverse-mode AD tape.
 *
 * Selects which forward computation and backward gradient rule ad_node_t
 * represents: constants/variables, elementwise arithmetic and math
 * functions, neural-network activation and tensor/transformer operations,
 * qLLM hyperbolic-geometry operations, and their scalar/tensor variants.
 * AD_NODE_TYPE_COUNT is a sentinel for bounds-checked dispatch tables and is
 * not itself a valid node type.
 */
typedef enum {
    // Core operations (0-11)
    AD_NODE_CONSTANT,
    AD_NODE_VARIABLE,
    AD_NODE_ADD,
    AD_NODE_SUB,
    AD_NODE_MUL,
    AD_NODE_DIV,
    AD_NODE_SIN,
    AD_NODE_COS,
    AD_NODE_EXP,
    AD_NODE_LOG,
    AD_NODE_POW,
    AD_NODE_NEG,

    // Activation gradients (12-18)
    AD_NODE_RELU,
    AD_NODE_SIGMOID,
    AD_NODE_SOFTMAX,
    AD_NODE_TANH,
    AD_NODE_GELU,
    AD_NODE_LEAKY_RELU,
    AD_NODE_SILU,

    // Tensor operation gradients (19-28)
    AD_NODE_CONV2D,
    AD_NODE_MAXPOOL2D,
    AD_NODE_AVGPOOL2D,
    AD_NODE_BATCHNORM,
    AD_NODE_LAYERNORM,
    AD_NODE_MATMUL,
    AD_NODE_TRANSPOSE,
    AD_NODE_RESHAPE,
    AD_NODE_SUM,
    AD_NODE_MEAN,

    // Transformer gradients (29-32)
    AD_NODE_ATTENTION,
    AD_NODE_MULTIHEAD_ATTENTION,
    AD_NODE_POSITIONAL_ENCODING,
    AD_NODE_EMBEDDING,

    // qLLM Geometric gradients (33-40)
    AD_NODE_HYPERBOLIC_DISTANCE,
    AD_NODE_POINCARE_EXP_MAP,
    AD_NODE_POINCARE_LOG_MAP,
    AD_NODE_TANGENT_PROJECT,
    AD_NODE_GEODESIC_ATTENTION,
    AD_NODE_MOBIUS_ADD,
    AD_NODE_MOBIUS_MATMUL,
    AD_NODE_GYROVECTOR_SPACE,

    // Additional math operations (41-44)
    AD_NODE_SQRT,
    AD_NODE_ABS,
    AD_NODE_SQUARE,
    AD_NODE_MAX,
    AD_NODE_MIN,

    // Phase 4 activation gradients (46-53)
    AD_NODE_ELU = 46,
    AD_NODE_SELU,
    AD_NODE_MISH,
    AD_NODE_HARDSWISH,
    AD_NODE_HARDSIGMOID,
    AD_NODE_SOFTPLUS,
    AD_NODE_DROPOUT,
    AD_NODE_CELU,

    // Complete math function gradients (54-66)
    AD_NODE_TAN = 54,
    AD_NODE_ASIN,
    AD_NODE_ACOS,
    AD_NODE_ATAN,
    AD_NODE_SINH,
    AD_NODE_COSH,
    AD_NODE_ASINH,
    AD_NODE_ACOSH,
    AD_NODE_ATANH,
    AD_NODE_LOG10,
    AD_NODE_LOG2,
    AD_NODE_EXP2,
    AD_NODE_CBRT,

    // Tensor AD nodes for qLLM bridge (67-79)
    AD_NODE_TENSOR_MATMUL = 67,       // dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    AD_NODE_TENSOR_SOFTMAX,            // Jacobian-vector product
    AD_NODE_TENSOR_LAYERNORM,          // Chain rule through mean/variance
    AD_NODE_TENSOR_RMSNORM,            // Chain rule through RMS
    AD_NODE_TENSOR_ATTENTION,          // 5-step chain rule through scaled dot-product
    AD_NODE_TENSOR_GELU,               // GELU backward
    AD_NODE_TENSOR_SILU,               // SiLU/Swish backward
    AD_NODE_TENSOR_TRANSPOSE,          // Permutation backward
    AD_NODE_TENSOR_SUM,                // Broadcast backward
    AD_NODE_TENSOR_BROADCAST_ADD,      // Sum-reduce backward
    AD_NODE_TENSOR_BROADCAST_MUL,      // Product-rule backward
    AD_NODE_TENSOR_EMBEDDING,          // Sparse update backward
    AD_NODE_TENSOR_CROSS_ENTROPY,      // Numerically stable backward
    AD_NODE_FRECHET_MEAN,              // Riemannian center of mass backward
    AD_NODE_ATAN2,                     // Binary atan2(y, x) backward

    // Sentinel for bounds checking
    AD_NODE_TYPE_COUNT
} ad_node_type_t;

/**
 * @brief One node of the reverse-mode AD computational graph.
 *
 * Records a single operation's forward value, accumulated backward
 * gradient, and links to its input nodes so the tape can be walked in
 * reverse topological order during backpropagation. Scalar operations use
 * `value`/`gradient`; tensor operations additionally use `tensor_value` /
 * `tensor_gradient` and, where a backward rule needs forward-pass
 * intermediates (softmax output, conv indices, etc.), `saved_tensors`.
 * `params` holds the small set of operation-specific scalars (e.g. conv
 * kernel/stride/padding, attention head count) needed to run the correct
 * backward rule for `type`.
 */
typedef struct ad_node {
    ad_node_type_t type;     // Type of operation
    double value;            // Computed value during forward pass (scalar)
    double gradient;         // Accumulated gradient during backward pass (scalar)
    struct ad_node* input1;  // First parent node (null for constants/variables)
    struct ad_node* input2;  // Second parent node (null for unary ops)
    size_t id;              // Unique node ID for topological sorting

    // === Extended fields for tensor operations ===
    // These are optional and only used for tensor-based AD nodes

    // Tensor data (when operating on tensors instead of scalars)
    void* tensor_value;      // Pointer to tensor value (null for scalar nodes)
    void* tensor_gradient;   // Pointer to tensor gradient (null for scalar nodes)

    // Additional inputs for multi-input operations (attention, etc.)
    struct ad_node* input3;  // Third input (e.g., V in attention)
    struct ad_node* input4;  // Fourth input (e.g., mask in attention)

    // Saved tensors for backward pass (e.g., softmax output, conv indices)
    void** saved_tensors;    // Array of saved tensors for backward
    size_t num_saved;        // Number of saved tensors

    // Operation parameters
    union {
        double alpha;        // For leaky_relu, dropout rate, etc.
        double curvature;    // For hyperbolic/Poincare operations
        int64_t axis;        // For reduction operations (sum, mean)
        struct {
            int64_t kernel_h, kernel_w;   // For conv2d, pooling
            int64_t stride_h, stride_w;
            int64_t pad_h, pad_w;
        } conv_params;
        struct {
            int64_t num_heads;            // For multi-head attention
            int64_t head_dim;
        } attention_params;
    } params;

    // Shape information for tensor operations
    int64_t* shape;          // Output shape
    size_t ndim;             // Number of dimensions
} ad_node_t;

/**
 * @brief Tape recording all AD nodes created during a forward pass.
 *
 * Nodes are appended to `nodes` in evaluation (creation) order as the
 * forward computation executes; backpropagation walks the tape in reverse.
 * `variables` tracks the subset of nodes that are independent input
 * variables, so gradients with respect to them can be extracted after the
 * backward pass completes.
 */
typedef struct ad_tape {
    ad_node_t** nodes;       // Array of nodes in evaluation order
    size_t num_nodes;        // Current number of nodes
    size_t capacity;         // Allocated capacity
    ad_node_t** variables;   // Input variable nodes
    size_t num_variables;    // Number of input variables
} ad_tape_t;


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_AD_H */
