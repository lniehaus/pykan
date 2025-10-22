#!/usr/bin/env python3
"""
Test script to verify the single coefficient mode of KANLayer
"""
import torch
import sys
import os

# Add the kan directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kan'))

from kan.KANLayer import KANLayer

def test_single_coef_mode():
    """Test the single coefficient mode functionality"""
    
    print("Testing KANLayer with single coefficient mode...")
    
    # Test parameters
    in_dim = 3
    out_dim = 2
    batch_size = 5
    
    # Create KANLayer with single coefficient mode
    layer = KANLayer(
        in_dim=in_dim, 
        out_dim=out_dim, 
        single_coef_mode=True,
        noise_scale=1.0
    )
    
    # Verify the layer configuration
    print(f"Layer configuration:")
    print(f"  in_dim: {layer.in_dim}")
    print(f"  out_dim: {layer.out_dim}")
    print(f"  num: {layer.num}")
    print(f"  k: {layer.k}")
    print(f"  single_coef_mode: {layer.single_coef_mode}")
    print(f"  coef shape: {layer.coef.shape}")
    
    # Verify that k=0 and num=1
    assert layer.k == 0, f"Expected k=0, got k={layer.k}"
    assert layer.num == 1, f"Expected num=1, got num={layer.num}"
    assert layer.coef.shape == (in_dim, out_dim, 1), f"Expected coef shape {(in_dim, out_dim, 1)}, got {layer.coef.shape}"
    
    # Create test input
    x = torch.randn(batch_size, in_dim)
    print(f"Input shape: {x.shape}")
    print(f"Input data:\n{x}")
    
    # Forward pass
    y, preacts, postacts, postspline = layer(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Expected output shape: {(batch_size, out_dim)}")
    print(f"Output data:\n{y}")
    
    # Verify output shapes
    assert y.shape == (batch_size, out_dim), f"Expected output shape {(batch_size, out_dim)}, got {y.shape}"
    assert preacts.shape == (batch_size, out_dim, in_dim), f"Expected preacts shape {(batch_size, out_dim, in_dim)}, got {preacts.shape}"
    assert postacts.shape == (batch_size, out_dim, in_dim), f"Expected postacts shape {(batch_size, out_dim, in_dim)}, got {postacts.shape}"
    assert postspline.shape == (batch_size, out_dim, in_dim), f"Expected postspline shape {(batch_size, out_dim, in_dim)}, got {postspline.shape}"
    
    # Manually verify the computation for single coefficient mode
    # The output should be: sum over input dimensions of (scale_base * base_fun(x) + scale_sp * coef * x) * mask
    base = layer.base_fun(x)  # (batch, in_dim)
    expected_spline = x[:, :, None] * layer.coef[None, :, :, 0]  # (batch, in_dim, out_dim)
    expected_y = layer.scale_base[None,:,:] * base[:,:,None] + layer.scale_sp[None,:,:] * expected_spline
    expected_y = layer.mask[None,:,:] * expected_y
    expected_y = torch.sum(expected_y, dim=1)  # (batch, out_dim)
    
    print(f"Manual computation result:\n{expected_y}")
    print(f"Difference from layer output: {torch.max(torch.abs(y - expected_y)).item()}")
    
    # Verify the results match (within floating point precision)
    assert torch.allclose(y, expected_y, atol=1e-6), "Manual computation doesn't match layer output"
    
    print("✓ Single coefficient mode test passed!")
    
    # Test grid update methods (should be no-ops)
    print("\nTesting grid update methods...")
    layer.update_grid_from_samples(x)
    print("✓ update_grid_from_samples completed (should be no-op)")
    
    # Test parent initialization (should be no-op)
    parent_layer = KANLayer(in_dim=in_dim, out_dim=out_dim, num=5, k=3)
    layer.initialize_grid_from_parent(parent_layer, x)
    print("✓ initialize_grid_from_parent completed (should be no-op)")
    
    # Test subset functionality
    print("\nTesting subset functionality...")
    in_subset = [0, 2]
    out_subset = [1]
    subset_layer = layer.get_subset(in_subset, out_subset)
    
    assert subset_layer.in_dim == len(in_subset), f"Expected subset in_dim {len(in_subset)}, got {subset_layer.in_dim}"
    assert subset_layer.out_dim == len(out_subset), f"Expected subset out_dim {len(out_subset)}, got {subset_layer.out_dim}"
    assert subset_layer.single_coef_mode == True, "Subset layer should inherit single_coef_mode"
    assert subset_layer.coef.shape == (len(in_subset), len(out_subset), 1), f"Expected subset coef shape {(len(in_subset), len(out_subset), 1)}, got {subset_layer.coef.shape}"
    
    print("✓ Subset functionality test passed!")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("Single coefficient mode is working correctly.")
    print("="*50)

def test_comparison_with_normal_mode():
    """Compare single coefficient mode with normal mode for simple cases"""
    
    print("\nTesting comparison between single coefficient and normal modes...")
    
    in_dim = 2
    out_dim = 1
    batch_size = 3
    
    # Create identical layers except for the mode
    layer_single = KANLayer(
        in_dim=in_dim, 
        out_dim=out_dim, 
        single_coef_mode=True,
        scale_base_mu=0.0,  # Disable base function for cleaner comparison
        scale_base_sigma=0.0
    )
    
    layer_normal = KANLayer(
        in_dim=in_dim, 
        out_dim=out_dim, 
        num=1,  # Single interval
        k=0,    # Degree 0 
        scale_base_mu=0.0,  # Disable base function for cleaner comparison
        scale_base_sigma=0.0
    )
    
    # Set the same coefficients manually for comparison
    test_coef = torch.tensor([[[0.5]], [[1.0]]], dtype=torch.float32)  # (in_dim, out_dim, 1)
    layer_single.coef.data = test_coef.clone()
    
    # For normal layer, we need to set the coefficients in the right format
    # Since k=0 and num=1, we should have one coefficient per input-output pair
    layer_normal.coef.data = test_coef.clone()
    
    # Test input
    x = torch.tensor([[1.0, 2.0], [0.5, -1.0], [-2.0, 3.0]], dtype=torch.float32)
    
    print(f"Test input:\n{x}")
    
    # Forward pass
    y_single, _, _, _ = layer_single(x)
    y_normal, _, _, _ = layer_normal(x)
    
    print(f"Single coef mode output:\n{y_single}")
    print(f"Normal mode output:\n{y_normal}")
    print(f"Difference: {torch.max(torch.abs(y_single - y_normal)).item()}")
    
    # The outputs should be very similar (but may not be exactly the same due to grid differences)
    print("✓ Comparison test completed!")

if __name__ == "__main__":
    test_single_coef_mode()
    test_comparison_with_normal_mode()
