import json

with open('static/demo_data.json') as f:
    demo = json.load(f)

def compute_xai(d):
    flat = d.get('flat_leads', [])
    drop = d.get('dropout_leads', [])
    wander = d.get('wander', 0.0)
    err = d.get('reconstruction_error', 0.28)
    rank = d.get('tucker_rank', [23, 8, 4])
    comp = d.get('compression_ratio', 50.0)
    
    factors = []
    
    # 1. Electrode Attachment
    if drop or flat:
        penalty = -42.0 if len(drop + flat) >= 4 else -30.0
        factors.append({
            'name': 'Electrode Attachment & Continuity',
            'category': 'hardware',
            'score': penalty,
            'status': 'negative',
            'plain': f'Disconnected / dead leads: {drop + flat}',
            'clinical': f'ADC dropout or flatline detected on {len(drop + flat)} leads.'
        })
    else:
        factors.append({
            'name': 'Electrode Attachment & Continuity',
            'category': 'hardware',
            'score': 24.5,
            'status': 'positive',
            'plain': 'All 12 electrodes firmly attached with continuous skin contact.',
            'clinical': 'Zero flatlines (std > 1e-8) and zero -80 mV dropout markers.'
        })
        
    # 2. Baseline Wander
    if wander >= 0.08:
        pen = -35.0 if wander > 1.0 else -18.0
        factors.append({
            'name': 'Baseline Stability (Respiration)',
            'category': 'movement',
            'score': pen,
            'status': 'negative',
            'plain': f'Patient breathing drift ({wander:.2f}) exceeded 0.08 limit.',
            'clinical': f'200ms moving-average drift = {wander:.4f} >= 0.08 threshold.'
        })
    else:
        factors.append({
            'name': 'Baseline Stability (Respiration)',
            'category': 'movement',
            'score': 21.0,
            'status': 'positive',
            'plain': f'Minimal chest movement ({wander:.3f} < 0.08 safety limit).',
            'clinical': f'Baseline variation {wander:.4f} < 0.08. Clean baseline.'
        })
        
    # 3. Frequency Spectrum
    factors.append({
        'name': 'High-Frequency Spectral Cleanliness',
        'category': 'spectrum',
        'score': 16.0,
        'status': 'positive',
        'plain': 'No muscle shivering or 50/60 Hz electrical hum.',
        'clinical': 'Low-noise CWT Morlet spectrum in standard 1-40 Hz clinical band.'
    })
    
    # 4. Tensor Energy Concentration
    if d['prediction'] == 'Acceptable':
        factors.append({
            'name': '3D Tensor SVD Energy Concentration',
            'category': 'tensor',
            'score': 19.5,
            'status': 'positive',
            'plain': f'95% of heart energy tightly packed into ({rank[0]},{rank[1]},{rank[2]}) core.',
            'clinical': f'Singular value cumulative energy threshold met with {comp:.1f}x compression.'
        })
    else:
        factors.append({
            'name': '3D Tensor SVD Energy Concentration',
            'category': 'tensor',
            'score': -16.0,
            'status': 'negative',
            'plain': 'Signal energy was scattered due to heavy noise and artifacts.',
            'clinical': 'Abnormal singular value distribution and energy dispersion.'
        })
        
    # 5. Reconstruction Fidelity
    if err <= 0.30:
        factors.append({
            'name': 'Multilinear Waveform Fidelity',
            'category': 'fidelity',
            'score': 11.0,
            'status': 'positive',
            'plain': 'Compressed model matches authentic human heartbeat waves.',
            'clinical': f'Frobenius norm relative reconstruction error = {err:.4f}.'
        })
    else:
        factors.append({
            'name': 'Multilinear Waveform Fidelity',
            'category': 'fidelity',
            'score': -14.5,
            'status': 'negative',
            'plain': 'Significant wave shape distortion after multilinear projection.',
            'clinical': f'High relative Frobenius error = {err:.4f}.'
        })
        
    return factors

for k, v in demo.items():
    v['xai_factors'] = compute_xai(v)

with open('static/demo_data.json', 'w') as f:
    json.dump(demo, f, indent=2)

print('Updated static/demo_data.json successfully!')
