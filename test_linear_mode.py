#!/usr/bin/env python3
"""
Test script to verify that the linear mode in KANLayer works correctly.
"""

import torch
import numpy as np
from kan.KANLayer import KANLayer

def test_linear_mode():
    """Test the linear mode functionality of KANLayer"""
    print("Testing KANLayer linear mode...")
    
    # Test parameters
    batch_size = 10
    in_dim = 3
    out_dim = 2
    
    # Create test data
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_dim)
    
    # Create KANLayer in linear mode
    kan_linear = KANLayer(
        in_dim=in_dim,
        out_dim=out_dim,
        mode='linear',
        scale_base_mu=0.0,  # Set to 0 to isolate linear effect
        scale_base_sigma=0.0  # Set to 0 to isolate linear effect
    )
    
    # Create KANLayer in default mode for comparison
    kan_default = KANLayer(
        in_dim=in_dim,
        out_dim=out_dim,
        mode='default',
        scale_base_mu=0.0,
        scale_base_sigma=0.0
    )
    
    print(f"Linear mode initialized: {kan_linear.mode}")
    print(f"Default mode initialized: {kan_default.mode}")
    
    # Test forward pass
    with torch.no_grad():
        y_linear, preacts_linear, postacts_linear, postspline_linear = kan_linear(x)
        y_default, preacts_default, postacts_default, postspline_default = kan_default(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Linear output shape: {y_linear.shape}")
    print(f"Default output shape: {y_default.shape}")
    
    # Verify that linear mode produces expected output shape
    assert y_linear.shape == (batch_size, out_dim), f"Expected shape ({batch_size}, {out_dim}), got {y_linear.shape}"
    assert postspline_linear.shape == (batch_size, out_dim, in_dim), f"Expected postspline shape ({batch_size}, {out_dim}, {in_dim}), got {postspline_linear.shape}"
    
    # Verify that linear weights exist and have correct shape
    assert hasattr(kan_linear, 'linear_weights'), "Linear weights should exist in linear mode"
    assert kan_linear.linear_weights.shape == (in_dim, out_dim), f"Linear weights should have shape ({in_dim}, {out_dim}), got {kan_linear.linear_weights.shape}"
    assert kan_linear.linear_weights.requires_grad, "Linear weights should be trainable"
    
    # Test that linear mode behaves like simple multiplication
    # Manually compute expected output
    with torch.no_grad():
        # The formula is: y = scale_base * base(x) + scale_sp * linear_weights * x
        # Since we set scale_base to 0, it should be: y = scale_sp * linear_weights * x
        expected_spline = x[:, :, None] * kan_linear.linear_weights[None, :, :]  # (batch, in_dim, out_dim)
        expected_y = torch.sum(kan_linear.scale_sp[None, :, :] * expected_spline, dim=1)  # (batch, out_dim)
        
        # Compare with actual output
        torch.testing.assert_close(y_linear, expected_y, rtol=1e-5, atol=1e-6)
    
    print("✓ Forward pass test passed")
    
    # Test that grid update functions don't crash in linear mode
    kan_linear.update_grid_from_samples(x)
    kan_linear.initialize_grid_from_parent(kan_default, x)
    print("✓ Grid update functions work correctly in linear mode")
    
    # Test subset functionality
    subset_layer = kan_linear.get_subset([0, 2], [1])
    assert subset_layer.mode == 'linear', "Subset should preserve linear mode"
    assert subset_layer.linear_weights.shape == (2, 1), f"Subset linear weights should have shape (2, 1), got {subset_layer.linear_weights.shape}"
    print("✓ Subset functionality works correctly in linear mode")
    
    # Test swap functionality
    original_weights = kan_linear.linear_weights.clone()
    kan_linear.swap(0, 1, mode='in')
    # Check that weights were swapped
    assert torch.allclose(kan_linear.linear_weights[0], original_weights[1]), "Weights should be swapped"
    assert torch.allclose(kan_linear.linear_weights[1], original_weights[0]), "Weights should be swapped"
    print("✓ Swap functionality works correctly in linear mode")
    
    # Test that gradients flow correctly
    kan_linear.train()
    optimizer = torch.optim.SGD(kan_linear.parameters(), lr=0.01)
    
    # Simple training step
    y_pred, _, _, _ = kan_linear(x)
    target = torch.randn_like(y_pred)
    loss = torch.nn.functional.mse_loss(y_pred, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check that linear weights have gradients
    assert kan_linear.linear_weights.grad is not None, "Linear weights should have gradients"
    print("✓ Gradient flow test passed")
    
    print("All tests passed! Linear mode is working correctly.")

def test_comparison_with_manual_implementation():
    """Compare KANLayer linear mode with manual linear layer"""
    print("\nComparing with manual linear implementation...")
    
    batch_size = 5
    in_dim = 4
    out_dim = 3
    
    torch.manual_seed(123)
    x = torch.randn(batch_size, in_dim)
    
    # Create KANLayer in linear mode with no base function contribution
    kan_linear = KANLayer(
        in_dim=in_dim,
        out_dim=out_dim,
        mode='linear',
        scale_base_mu=0.0,
        scale_base_sigma=0.0
    )
    
    # Create manual linear transformation with same weights
    with torch.no_grad():
        manual_weights = kan_linear.linear_weights.clone()
        manual_scale_sp = kan_linear.scale_sp.clone()
    
    # Forward pass through KANLayer
    y_kan, _, _, _ = kan_linear(x)
    
    # Manual computation
    with torch.no_grad():
        # KANLayer computes: sum over input dims of (scale_sp * linear_weights * x)
        manual_output = torch.sum(manual_scale_sp[None, :, :] * x[:, :, None] * manual_weights[None, :, :], dim=1)
    
    # They should be identical
    torch.testing.assert_close(y_kan, manual_output, rtol=1e-6, atol=1e-7)
    print("✓ KANLayer linear mode matches manual implementation")

if __name__ == "__main__":
    test_linear_mode()
    test_comparison_with_manual_implementation()
    print("\n🎉 All tests completed successfully!")
