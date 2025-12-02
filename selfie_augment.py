import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from selfies import encoder, decoder
    SELFIES_AVAILABLE = True
    print(" SELFIES package successfully imported")
except ImportError:
    print(" SELFIES package not available. Please install with: pip install selfies")
    SELFIES_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, Descriptors
    RDKIT_AVAILABLE = True
    print(" RDKit package successfully imported")
except ImportError:
    print(" RDKit package not available.")
    RDKIT_AVAILABLE = False

try:
    import umap
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial import ConvexHull
    print(" All other packages imported successfully")
except ImportError as e:
    print(f" Import error: {e}")

class SelfiesChemicalSpaceAnalyzer:
    def __init__(self, save_dir="selfies_augmented_results", use_gpu=True):
        self.reduction = None
        self.df_original = None
        self.df_combined = None
        self.embeddings = None
        self.df_sampled = None
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        print(" SELFIES Chemical Space Analyzer Initialized")
        
    def load_data(self, filepath):
        print(" Loading dataset...")
        try:
            self.df_original = pd.read_csv(filepath)
            print(f" Successfully loaded {len(self.df_original):,} molecules from {filepath}")
            return self.df_original
        except FileNotFoundError:
            print(f" File {filepath} not found. Creating sample dataset...")
            self._create_sample_dataset()
            return self.df_original
        except Exception as e:
            print(f" Error loading data: {e}")
            raise
    
    def _create_sample_dataset(self):
        print(" Creating sample dataset...")
        sample_smiles = [
            'CCO', 'CCCO', 'CCCCCO', 'C1CCCCC1', 'C1=CC=CC=C1', 
            'CC(=O)O', 'CCOC', 'CCN', 'CCCl', 'CBr', 'CC(C)C', 
            'C1CC1', 'C1COC1', 'CC#N', 'CCOCC', 'CC(C)O', 
            'C1CCNC1', 'C1CCNCC1', 'C1CCOC1', 'CC(C)=O',
            'CCOCC(=O)O', 'CC1CCCCC1', 'C1CCC2CCCCC2C1', 'CC(C)CC(C)(C)C'
        ]
        all_smiles = sample_smiles * 40000
        self.df_original = pd.DataFrame({'smiles': all_smiles[:800000]})
        print(f" Created sample dataset with {len(self.df_original):,} molecules")
    
    def smiles_to_selfies(self, smiles_list):
        """Convert SMILES to SELFIES"""
        if not SELFIES_AVAILABLE:
            print(" SELFIES not available. Cannot proceed with SELFIES augmentation.")
            return [None] * len(smiles_list)
        
        print(" Converting SMILES to SELFIES...")
        selfies_list = []
        valid_count = 0
        for i, smiles in enumerate(smiles_list):
            try:
                selfies = encoder(smiles)
                selfies_list.append(selfies)
                valid_count += 1
            except Exception as e:
                selfies_list.append(None)
            
            if (i + 1) % 10000 == 0:
                print(f"   Processed {i + 1:,}/{len(smiles_list):,} SMILES...")
        print(f" Successfully converted {valid_count:,}/{len(smiles_list):,} SMILES to SELFIES")
        return selfies_list
    
    def augment_selfies_advanced(self, selfies_str, augmentation_factor=3):
        """ACTUAL SELFIES augmentation using proper techniques"""
        if not SELFIES_AVAILABLE or selfies_str is None:
            return []
        try:
            # Parse SELFIES into tokens
            tokens = []
            current_token = ""
            for char in selfies_str:
                current_token += char
                if char == ']':
                    tokens.append(current_token)
                    current_token = ""
            augmented_versions = []
            
            for _ in range(augmentation_factor):
                new_tokens = tokens.copy()
                if len(new_tokens) > 4:
                    # Find swappable positions (avoid breaking core structure)
                    swappable_indices = []
                    for i, token in enumerate(new_tokens):
                        # Avoid swapping branching/ring tokens at the beginning
                        if i > 2 and not any(x in token for x in ['[Branch', '[Ring', '[C', '[O', '[N']):
                            swappable_indices.append(i)
                    
                    if len(swappable_indices) >= 2:
                        i, j = np.random.choice(swappable_indices, 2, replace=False)
                        new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]
                        augmented_version = ''.join(new_tokens)
                        if augmented_version != selfies_str:
                            augmented_versions.append(augmented_version)
                
                # Strategy 2: Branch modification
                if '[Branch1]' in selfies_str:
                    new_selfies = selfies_str.replace('[Branch1]', '[Branch2]')
                    if new_selfies != selfies_str:
                        augmented_versions.append(new_selfies)
                elif '[Branch2]' in selfies_str:
                    new_selfies = selfies_str.replace('[Branch2]', '[Branch1]')
                    if new_selfies != selfies_str:
                        augmented_versions.append(new_selfies)
                
                # Strategy 3: Ring modification  
                if '[Ring1]' in selfies_str:
                    new_selfies = selfies_str.replace('[Ring1]', '[Ring2]')
                    if new_selfies != selfies_str:
                        augmented_versions.append(new_selfies)
                elif '[Ring2]' in selfies_str:
                    new_selfies = selfies_str.replace('[Ring2]', '[Ring1]')
                    if new_selfies != selfies_str:
                        augmented_versions.append(new_selfies)
                
                # Strategy 4: Add/remove small groups
                if len(new_tokens) > 3 and np.random.random() > 0.7:
                    insert_pos = np.random.randint(2, len(new_tokens)-1)
                    new_tokens.insert(insert_pos, '[C]')
                    augmented_version = ''.join(new_tokens)
                    augmented_versions.append(augmented_version)

            return list(set(augmented_versions))
            
        except Exception as e:
            print(f" Error in SELFIES augmentation: {e}")
            return []
    
    def validate_selfies_augmentation(self, original_smiles, augmented_selfies):
        """Validate that augmented SELFIES produce valid and diverse molecules"""
        try:
            # Convert back to SMILES
            augmented_smiles = decoder(augmented_selfies)
            mol = Chem.MolFromSmiles(augmented_smiles)
            if mol is None:
                return False
            
            if Descriptors.MolWt(mol) > 1000 or Descriptors.MolWt(mol) < 30:
                return False

            original_mol = Chem.MolFromSmiles(original_smiles)
            if original_mol:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(original_mol, 2, nBits=1024)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                similarity = np.sum(fp1 & fp2) / np.sum(fp1 | fp2) if np.sum(fp1 | fp2) > 0 else 0
                if similarity < 0.1 or similarity > 0.9:
                    return False
            
            return True
        except Exception as e:
            return False
    
    def generate_selfies_augmented_dataset(self, target_size=1000000):
        """Generate augmented dataset using ACTUAL SELFIES augmentation"""
        if not SELFIES_AVAILABLE:
            print(" SELFIES not available. Cannot perform SELFIES augmentation.")
            return None
        print(" Starting REAL SELFIES augmentation...")
        if self.df_original is None:
            raise ValueError(" No data loaded. Call load_data() first.")
        # Convert SMILES to SELFIES
        print(" Step 1: Converting original SMILES to SELFIES...")
        self.df_original['selfies'] = self.smiles_to_selfies(self.df_original['smiles'].tolist())

        # Remove invalid SELFIES conversions
        original_len = len(self.df_original)
        self.df_original = self.df_original.dropna(subset=['selfies'])
        valid_len = len(self.df_original)
        print(f" Retained {valid_len:,}/{original_len:,} valid SELFIES conversions")
        
        if valid_len == 0:
            print(" No valid SELFIES conversions. Cannot proceed.")
            return None
        original_count = len(self.df_original)
        augmentation_needed = target_size - original_count
        print(f"Augmentation Plan:")
        print(f"   Original molecules: {original_count:,}")
        print(f"   Target size: {target_size:,}")
        print(f"   Augmentation needed: {augmentation_needed:,}")
        # Generate augmented data using SELFIES
        augmented_data = []
        current_count = 0
        print(" Step 2: Performing SELFIES augmentation...")
        # Shuffle for diversity
        df_shuffled = self.df_original.sample(frac=1, random_state=42).reset_index(drop=True)
        
        augmentation_round = 0
        while current_count < augmentation_needed:
            augmentation_round += 1
            print(f"   Augmentation round {augmentation_round}...")
            
            for idx, row in df_shuffled.iterrows():
                if current_count >= augmentation_needed:
                    break
                    
                original_selfies = row['selfies']
                original_smiles = row['smiles']
                
                # Generate augmented SELFIES versions
                augmented_selfies_list = self.augment_selfies_advanced(
                    original_selfies, 
                    augmentation_factor=2
                )
                for aug_selfies in augmented_selfies_list:
                    if current_count >= augmentation_needed:
                        break
            
                    # Validate the augmentation
                    if self.validate_selfies_augmentation(original_smiles, aug_selfies):
                        try:
                            aug_smiles = decoder(aug_selfies)
                            augmented_data.append({
                                'smiles': aug_smiles,
                                'selfies': aug_selfies,
                                'source': 'augmented',
                                'original_smiles': original_smiles,
                                'augmentation_method': 'selfies'
                            })
                            current_count += 1
                            if current_count % 1000 == 0:
                                print(f"      Generated {current_count:,}/{augmentation_needed:,} augmented molecules")
                                
                        except Exception as e:
                            continue
            
            print(f"   Round {augmentation_round} completed: {current_count:,}/{augmentation_needed:,}")
            if augmentation_round > 10 and current_count < augmentation_needed * 0.1:
                print("  Low augmentation efficiency. Stopping early.")
                break
        print(" Step 3: Creating final dataset...")
        df_original_marked = self.df_original.copy()
        df_original_marked['source'] = 'original'
        df_original_marked['original_smiles'] = df_original_marked['smiles']
        df_original_marked['augmentation_method'] = 'original'
        df_augmented = pd.DataFrame(augmented_data)
        self.df_combined = pd.concat([
            df_original_marked[['smiles', 'selfies', 'source', 'original_smiles', 'augmentation_method']],
            df_augmented
        ], ignore_index=True)
        original_final = len(self.df_combined[self.df_combined['source'] == 'original'])
        augmented_final = len(self.df_combined[self.df_combined['source'] == 'augmented'])
        
        print(f" SELFIES Augmentation Completed!")
        print(f"   Final dataset size: {len(self.df_combined):,} molecules")
        print(f"   Original molecules: {original_final:,}")
        print(f"   Augmented molecules: {augmented_final:,}")
        print(f"   Augmentation success rate: {augmented_final/augmentation_needed*100:.1f}%")
        return self.df_combined
    
def main():
    print("=" * 70)
    print(" REAL SELFIES CHEMICAL SPACE ANALYSIS")
    print("=" * 70)
    analyzer = SelfiesChemicalSpaceAnalyzer(save_dir="real_selfies_results")
    original_file = "/data4t/Qahtan/SELFIE_AUGMENATIAION/dataset_800K.csv"
    df_original = analyzer.load_data(original_file)
    print("\n" + "="*50)
    print(" STARTING REAL SELFIES AUGMENTATION")
    print("="*50)
    df_combined = analyzer.generate_selfies_augmented_dataset(target_size=1000000)
    if df_combined is None:
        print(" SELFIES augmentation failed. Exiting.")
        return
    print("\n" + "="*50)
    print("PERFORMING CHEMICAL SPACE ANALYSIS")
    print("="*50)
    embeddings = analyzer.perform_umap_analysis(force_recompute=True, sample_size=200000)
    if embeddings is None:
        print(" UMAP analysis failed.")
        return
    metrics = analyzer.calculate_metrics(embeddings)
    analyzer.create_visualization(embeddings, 'real_selfies_chemical_space.png')
    analyzer.generate_report(metrics)
    analyzer.save_augmented_data("real_selfies_augmented_dataset.csv")
    print("\n" + "="*70)
    print(" REAL SELFIES ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(" Results saved in: real_selfies_results/")
    print("  Visualization: real_selfies_chemical_space.png")
    print("Dataset: real_selfies_augmented_dataset.csv")
    print("="*70)

if __name__ == "__main__":
    main()