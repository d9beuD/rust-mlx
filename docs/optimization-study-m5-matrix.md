# Unités matricielles M5 et Metal 4.1 : résultats sur le décode solo

Sur le M5 Max40/128Go, macOS27.0.1 et MLX0.32.2, **aucun des nouveaux chemins ne mérite d'être activé par défaut**. Le contrôle final du moteur conserve les256 IDs canoniques dans quatre générations MTP, à une médiane de68,715tok/s. C'est un nouveau cohorte de contrôle, pas un gain attribuable aux prototypes. Les anciens45,28/68,43 plain/MTP restent leurs mesures historiques.100tok/s solo n'est pas atteint.

Cette suite donne des kernels Rust/Metal exécutables, un benchmark de207 configurations mesurées et les contre-exemples qui empêchent leur promotion. Elle complète [l'étude quantification/assembleur](optimization-study-quantization-metal.md).

## Ce qui est disponible sur macOS27

[Apple décrit les entrées coopératives, les formats quantifiés et les plans de scales](https://developer.apple.com/videos/play/wwdc2026/330/). Le backend MLX installé sélectionne déjà le langage Metal4.1 sur macOS27 : ces opérations passent par notre frontière MLX-C existante. Aucun fork, bridge C++ ni changement de dépendance n'a été nécessaire.

L'API installée contraint les deux entrées coopératives à M16/32, N16/32 et K16/32, avec au moins une dimension32. Deux ou quatre positions ne deviennent donc pas une petite tuile M4 par simple réglage du descripteur. Une entrée hybride permet M8/K64. Les tenseurs quantifiés inline utilisent un pointeur d'octets et un layout compact ; la quantification affine avec biais exige encore sa correction explicite.

La capacité physique d'un tenseur coopératif peut dépasser M×N/32. Le premier résultat Q4 compacté, dimensionné à huit floats par thread, corrompait la moitié des sorties. Le remplacement par une accumulation allouée par TensorOps corrige ce cas ; la version rejetée est conservée comme calibration, jamais comme preuve de sûreté. Les entrées physiques sont initialisées, et les types de destination correspondent aux types d'opérandes réels.

## Variantes exécutées

| Variante | Disposition | Résultat |
|---|---|---|
| Staged | M16/N32/K16, poids reconstruits BF16 en mémoire de threadgroup, split-K1/2/4/8 | Différences natives sur Q4/Q5/Q6/Q8 ; pas de gain qualifié |
| Registers | Même tuile, poids et activations en tenseurs coopératifs | Fonctionne sans instrumentation sur les fixtures ; contre-exemple numérique sous validation Metal, chemin sélectionné alors natif |
| Hybrid (`CompactRegisters`) | M8/N32/K64, activations device et poids en registres | Différences natives ; bénéfice isolé trop faible et non exact |
| Affine | Codes entiers BF16 exacts, scales/biais après le dot,32 partitions natives réparties sur8 ou32 SIMD groups | Premières projections Q4/Q5/Q6 exactes, mais lenteur et échec du vérificateur complet ; tête Q8 non exacte |
| Packed | Codes Q4 directement lus comme `uint4b_format`, M8/N32/K32 ; deux partitions natives représentées par deux lignes virtuelles par position | Projections testées exactes, vérificateurs courts et rollback exacts, mais divergence du test256 tokens |

Le benchmark utilise les vrais poids et les formes BF16 du checkpoint mixte : QKV et Z Q6/g64, Q Q4/g64, down HC Q5/g64 et tête Q8/g64. **Ses activations sont synthétiques** : l'oracle indépendant fait traverser la couche0 à sin(j×0,013) BF16[1,10,10240], puis enregistre ses étapes ; les autres modules réutilisent une étape de forme compatible. Il ne mesure ni qualité textuelle ni acceptation du brouillon. Les vérifications de modèle complet utilisent séparément les dix IDs de prompt canoniques.

## Mesures finales de projection

Exemple q_proj réelle12288×2560 à quatre positions, split8. Chaque ligne comporte son propre QMV de contrôle et30 paires alternées après cinq paires d'échauffement, sauf Packed100. Comparer les gains appariés, pas les baselines de cohorte différents. Les services StorageManagement/ApplicationsStorageExtension et WindowServer étaient actifs ; aucune compilation ou autre expérience GPU de cette tâche n'a été lancée pendant ces cohorte finaux.

| Variante | QMV médian µs | Candidate médiane µs | Gain médian apparié | Exacte sur cette entrée |
|---|---:|---:|---:|---|
| Staged | 253,42 | 244,35 | +2,37% | Non |
| Registers | 195,08 | 297,81 | −42,09% | Non |
| Hybrid | 356,33 | 341,96 | +2,74% | Non |
| Affine8 SIMD groups | 184,88 | 477,54 | −60,44% | Oui |
| Packed8 SIMD groups | 199,79 | 213,67 | −8,89% | Oui |

Packed reste également plus lent à deux et trois positions :−18,06% et−16,34% appariés. Les échantillons complets, CV et intervalles bootstrap sont dans [matrix-study-summary.json](../results/matrix-study-summary.json). Une erreur de poids de reconstruction BF16 affecte beaucoup plus d'éléments que le simple changement d'ordre de réduction ; les erreurs des deux familles sont conservées sans élargir la référence.

## Les tests complets changent la conclusion

Le premier chemin affine Q4/Q5/Q6 échoue dans son essai instrumenté à trois positions, après un cas à deux positions exact : erreurs maximales logits0,625, hidden0,234375 et états1,125. La précision d'une projection isolée ne se transpose pas à toute la chaîne récurrente.

Le chemin Packed passe les blocs2–8 deux fois, logits/hidden/GDN/KV/QSA/PLE et historique CPU compris. Seuls2–4 engagent le nouveau shader ;5–8 conservent le QMV borné natif. Les cinq préfixes de rollback0–4 et leur continuation passent aussi. Ces temps instrumentés ne sont pas du débit.

Le protocole long, avec échauffement256 puis quatre paires prévues, s'arrête dès la première candidate : **le token d'index103 passe de2830 à1048**. Le rejeu strict enregistre les256 IDs produits, les256 attendus, les propositions, l'acceptation et3071 appels de projection. Son rapport est `complete=false`, `qualified=false`. Ce rejeu a chevauché des diagnostics portables et ses temps ne sont pas qualifiés ; la première tentative sans ces diagnostics échouait déjà. Les paires suivantes et les cohorte chat ne sont pas mesurés comme gains d'une candidate qui échoue la trajectoire.

Le chemin à deux entrées coopératives et accumulation matricielle a une autre limite mesurée sous validation Metal : BF16 Q4/g32, T1, sortie0 au lieu de0,006225586. Un cas dense différencie également staged et registers. Ni les types d'opérande explicites ni l'initialisation des registres physiques n'ont résolu ce problème. Sa cause n'est pas établie ; ce n'est pas une preuve attribuant un bug au matériel, à MLX ou au SDK. Le chemin affine à produits séparés et l'entrée hybride ont aussi été exécutés sur les formes réelles sous validation : leur résultat ne suffit pas à généraliser le contre-exemple à toutes les opérations coopératives. Le test reproduit ce contre-exemple, puis exige le repli natif de la géométrie sélectionnée. Une suite passant ce test négatif ne certifie pas ce kernel comme exact.

## Dispatch et qualification

Les variantes sont limitées au M5/macOS27, avec garde sur dtype, bits/groupe, dimensions et indexation. Elles restent désactivées, sans variable d'environnement activant le serveur. Les modes de compilation HC/GDN comprennent les contrôles de dispatch dans leur clé. Batching indépendant et blocs longs conservent leur fallback. Les drapeaux des CLI de recherche permettent de reproduire les rejets, pas d'annoncer une optimisation stable.

Le code final passe format, Clippy strict et29 tests release. La suite instrumentée passe28 tests portables ; elle distingue expressément les cas exacts staged/packed du contre-exemple coopératif et de son fallback. Le vérificateur natif complet, Packed court et son rollback sont recontrôlés avec abort-on-fault. Les rapports de composant et instrumentés restent séparés de `performance-summary.json`. Les empreintes des sources, rapports et qualifications HTTP sont dans [matrix-study-validation.json](../results/matrix-study-validation.json). Le serveur passe FIFO et batch8, streaming, annulation et64 comparaisons Unicode par mode ; les serveurs de test sont arrêtés.

## Reproduction

```sh
PATH="$PWD/.venv/bin:$PATH" scripts/check.sh
PATH="$PWD/.venv/bin:$PATH" scripts/validate-metal.sh
target/release/matrix-bench --model TARGET --output results/matrix-final-staged-register-components.json --repetitions 30
target/release/matrix-bench --model TARGET --output results/matrix-final-hybrid-components.json --compact --repetitions 30
target/release/matrix-bench --model TARGET --output results/matrix-final-affine-components.json --affine --compact --repetitions 30
target/release/matrix-bench --model TARGET --output results/matrix-final-packed-components.json --packed --repetitions 100
target/release/verify-parity --matrix-packed --output results/matrix-final-packed-verifier-metal.json
target/release/rollback-parity --matrix-packed --output results/matrix-final-packed-rollback-metal.json
target/release/mtp-infer --model TARGET --ab-kernel matrix-packed --max-tokens 256 --warmup-tokens 256 --runs 1 --draft-depth 3 --ignore-eos --expected results/target-baseline-256.json --output results/matrix-packed-rejected-256.json
target/release/mtp-infer --model TARGET --max-tokens 256 --warmup-tokens 256 --runs 4 --draft-depth 3 --ignore-eos --expected results/target-baseline-256.json --output results/matrix-final-default-mtp-256.json
.venv/bin/python scripts/analyze_matrix_study.py
```

`TARGET` est `/Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp`. Pour les vérificateurs/rollback instrumentés, préfixer la commande des quatre variables `MTL_SHADER_VALIDATION=1`, `MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING=1`, `MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`, `MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1`. La commande de rejeu long doit échouer ; l'analyse vérifie le rejet et refuse son contrôle négatif sans différence d'IDs.

MTPLX et oMLX restent des inspirations de disposition, décrites avec leurs commits dans l'étude précédente. Les gains de leurs kernels prefill/activations INT8 ne deviennent pas un gain exact de notre décode MTP court. Pour exploiter davantage les unités matricielles, il faudrait soit plus de lignes utiles par poids/expert, soit un format de checkpoint adapté et calibré, soit un algorithme préservant l'ordre natif des réductions à moindre coût. Aucun de ces bénéfices n'est démontré ici.
