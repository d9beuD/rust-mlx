# Étude : quantification, Metal et assembleur pour le décode solo

Étude du 6 octobre 2026, sur le checkpoint local `Qwen3.8-Flash-Next-oQ4e-mtp`, M5 Max 40 cœurs GPU, 128 Go, macOS 27.0.1, MLX 0.32.2. Base du moteur : `e1ca7024ab9d892adb8e5ff327ee049a8c74239b`. Cet inventaire et ces probes n'ont modifié aucun poids du checkpoint. Les prototypes Rust/Metal réalisés ensuite et leurs rejets sont détaillés dans [l'étude M5](optimization-study-m5-matrix.md) ; les réglages par défaut restent inchangés.

La mesure Rust qualifiée lors de cet inventaire était **68,43 tok/s en MTP solo** ; le contrôle ultérieur de l’étude M5 donne68,72tok/s, sans nouveau kernel activé. Les nouvelles mesures ci-dessous concernent une projection isolée dans l'oracle Python. Elles ne constituent ni un nouveau débit du moteur, ni une validation de qualité du modèle.

Critères de cette étude : inspecter les sources des deux moteurs à des révisions identifiées ; compter les octets du checkpoint réel ; tester la faisabilité des formats et de l'assembleur avec les outils installés ; distinguer les modifications exactes des approximations ; proposer des expériences et leurs conditions de validation. Ces critères sont satisfaits par les deux rapports reproductibles liés ci-dessous.

## Ce que contient réellement le checkpoint

L'inventaire lit uniquement les en-têtes des 21 shards Safetensors. Les nombres sont des Go décimaux et incluent les métadonnées de quantification lorsqu'elles existent.

| Ensemble | Taille des tenseurs | Lecture utile au décode |
|---|---:|---|
| Experts routés du modèle cible | 67,95 Go | 10 experts parmi 512 par couche ; tous sont affine Q4/g64 |
| Autres poids du modèle cible, hors tête et embedding | 2,60 Go | Mélange Q4/Q5/Q6/Q8 et petits tenseurs non quantifiés |
| Tête de sortie | 675,43 Mo | Q8/g64 ; vocabulaire complet 248 320 × 2560 |
| Embedding | 675,43 Mo | Lecture de quelques lignes, pas de balayage complet |
| Bloc MTP privé | 1,48 Go | Majoritairement Q4, avec protections Q5/Q6/Q8 |
| Table PLE | 32,00 Go | Q4/g32 ; lignes mmap CPU, pas 32 Go lus par token |
| Vision | 0,90 Go | Hors périmètre texte |

Le nom « oQ4e » ne signifie donc pas que chaque module est en 4 bits. Le modèle emploie **652 overrides de quantification** ; les normes et petits paramètres ont encore leur précision propre.

Un modèle comptable, qui suppose une lecture de chaque projection résidente et de chaque expert choisi, donne **4,467 Go par position cible** : 2,464 Go de projections résidentes quantifiées, 1,327 Go d'experts choisis et 0,675 Go de tête. Il exclut cache KV, états GDN, PLE, réutilisation en cache matériel et lectures répétées. Ce n'est pas une mesure DRAM, et ce n'est pas le coût d'un bloc MTP de quatre positions.

Conséquence utile : requantifier seulement les projections résidentes en affine Q4, avec leurs groupes actuels et métadonnées BF16, économiserait environ **467 Mo**, soit **10,46 %** de ces octets idéalisés. Même avec un décode entièrement limité par ces lectures, cela représenterait environ 11,7 % de débit supplémentaire, sous les hypothèses de ce modèle. Aucun gain réel n'est encore mesuré pour cette conversion.

## Quantifier davantage le brouillon MTP

Cette piste préserve les poids du modèle cible. Dans notre API greedy, la vérification complète du modèle cible peut préserver les tokens émis même si les propositions du brouillon changent. Elle exige toujours les tests de commit/rollback des caches. Pour un futur sampler stochastique, il faudra aussi la correction probabiliste adaptée à la nouvelle distribution de proposition.

J'ai requantifié **une copie de la tête uniquement**, par morceaux de 8192 lignes, et testé les entrées MTP déjà enregistrées : dix positions de prefill et deux positions de decode. La tête Q8 de référence reproduit exactement les trois logits sauvegardés disponibles. Chaque projection de comparaison numérique est exécutée séparément en géométrie singleton.

| Copie de tête | Taille | Médiane de projection T1 | Argmax identique sur les 12 entrées | RMSE des logits |
|---|---:|---:|---:|---:|
| Q8 d'origine | 675,43 Mo | 1,35–1,38 ms selon le cohort apparié | 12/12 | 0 |
| Affine Q4/g64 | 357,58 Mo | 0,812 ms | 9/12 | 0,233 |
| Affine Q6/g64 | 516,51 Mo | 1,100 ms | 11/12 | 0,060 |
| MXFP4/g32 | 337,72 Mo | 0,786 ms | 9/12 | 0,295 |
| NVFP4/g16 | 357,58 Mo | 0,809 ms | 11/12 | 0,283 |

Les mesures comportent un échauffement puis 30 paires alternées par format et géométrie T1/T4 ; les échantillons, coefficients de variation et IDs sont conservés. Des services macOS concurrents étaient actifs, notamment StorageManagement et WindowServer. Les différences de quelques pourcents entre candidats ne permettent pas de les départager solidement.

**9/12 ou 11/12 n'est pas un taux d'acceptation MTP.** Ces entrées sont imposées, sur un seul prompt ; les trajectoires autoregressives du brouillon n'ont pas été recalculées. NVFP4 ne devient pas « aussi bon que Q6 » sur cette seule observation. La conversion repart des poids Q8 déquantifiés en BF16, et ajoute donc une seconde erreur de quantification.

Avec trois propositions, économiser approximativement 0,57 ms par tête ferait gagner environ 1,7 ms par round, si l'acceptation restait identique. Rapporté à un round historique d'environ 47 ms, cela suggère seulement 3–4 % de réduction de durée. C'est une estimation combinant deux expériences, pas un benchmark de décode. Une baisse d'acceptation peut l'annuler.

Le meilleur protocole suivant est de comparer une tête Q6 complète, une tête Q4 complète, puis une tête Q8 limitée à un vocabulaire fréquent de 32K/64K/96K, éventuellement combinée à Q6. Une liste de fréquence multilingue indépendante des benchmarks est préférable à la shortlist de 4096 IDs déjà rejetée. Il faut compter préparation, sélection, éventuels retours à la tête complète, acceptation et latence totale sur français, code, SQL et texte général.

## Autres formats de poids et d'activations

[MLX documente affine, MXFP4, MXFP8 et NVFP4](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html). Le backend installé 0.32.2 et les probes confirment la disponibilité des formats testés. Notre chargeur Rust accepte actuellement seulement affine ; les wrappers ordinaires de mlx-rs passent également ce mode explicitement. MLX-C expose déjà le paramètre de mode : une petite extension à cette frontière serait suffisante, sans fork MLX ni nouveau bridge C++.

| Transformation | Opportunité | Limite et priorité |
|---|---|---|
| Répartition de bits selon la sensibilité des modules | Baisser certains Q8/Q6/Q5 résidents, protéger routes, GDN et HC sensibles | Déjà partiellement réalisée par oQ4e ; variante de modèle à calibrer et évaluer, pas optimisation exacte |
| Experts Q4 → Q3 | Réduire les 67,95 Go de banques | Les banques sont lues sélectivement : économie idéale par position de 295 Mo, soit 6,60 % du modèle comptable total ; dépacking 3 bits plus coûteux et qualité à vérifier |
| Experts affine Q4/g64 → MXFP4/g32 | Scales compactes, éventuellement calcul TensorOps direct | 4,5 → 4,25 bits effectifs par poids, seulement 1,65 % d'économie du trafic total idéalisé ; le principal intérêt éventuel serait le kernel, pas la taille |
| Poids Q4/Q5 + activations INT8, dit W4A8/W5A8 | Multiplications INT8 × INT8 → INT32 sur unités matricielles | Quantification dynamique des activations, donc approximation ; frais de préparation potentiellement dominants à T1/T4 |
| KV en INT8/INT4 ou TurboQuant | Réduire mémoire et lecture d'attention à long contexte | Modifie les valeurs vues par le modèle cible ; priorité inférieure au décode court, qui a seulement deux têtes KV et de l'attention QSA sparse |
| Repacking sans perte des codes et métadonnées existants | Loads vectoriels, disposition des tuiles, gate/up adjacents | Préserve les valeurs stockées, mais le kernel doit aussi préserver les arrondis et réductions ; comparer coût de préparation et mémoire supplémentaire |

Changer de groupe, enlever le biais affine ou passer à un codebook FP4 ne constitue pas un simple changement de sérialisation. Ces transformations requantifient normalement les valeurs. Les quatre formats testés ici sont différents de formats GGUF portant eux aussi un nom « Q4 ».

Pour convertir le modèle cible avec une évaluation de qualité crédible, il faudrait idéalement ses poids sources en BF16/FP16 et un corpus de calibration indépendant. GPTQ/AWQ ou l'imatrix d'oQe servent à allouer/réduire l'erreur ; ils n'apportent pas automatiquement un kernel plus rapide. Il faut mesurer le couple format et kernel, et conserver ces variantes comme checkpoints distincts. Aucun téléchargement massif ou écrasement de ce checkpoint n'a été entrepris.

## Assembleur : quelles possibilités sont réelles ?

Le probe local avec `xcrun metal -std=metal4.0` accepte SIMD et la déclaration d'une opération `mpp::tensor_ops::matmul2d`, mais refuse `asm("nop")` avec **`illegal asm statement`**. Il ne s'agit pas d'une interface publique comparable à l'assembleur inline CUDA/PTX. Le probe MPP confirme la compilation des types, pas l'exécution d'un matmul ni sa performance.

La voie pratique consiste à guider le compilateur Metal : intrinsics comme `extract_bits`, vecteurs de chargement, déroulage spécialisé par bits/groupe, pression de registres, tuiles, barrières et réduction. Le décodeur Q4/Q5 d'oMLX fournit un exemple de décodage de champs alignés sans division, avec garde des accès à cheval sur deux mots. Sa revendication « une instruction » doit être vérifiée sur le code réellement généré et les compteurs du GPU, pas reprise comme une mesure locale.

Compiler en `.air`/`.metallib` peut réduire le démarrage et déplacer une partie de la compilation. Cela ne prouve pas que le shader exécuté en régime chaud sera meilleur. Le produit chauffe déjà ses kernels ; l'API Metal actuelle de MLX reste le premier chemin de prototypage.

Le CPU local annonce `FEAT_SME=1` et `FEAT_SME2=1`. NEON/SME2 permettent des optimisations CPU, avec des contraintes de mode streaming et d'ABI décrites dans [la documentation XNU d'Apple](https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md). SME2 CPU, accélérateurs matriciels du GPU et Apple Neural Engine sont des unités distinctes. Une réécriture assembleur des grands produits GPU n'est pas justifiée par cette disponibilité. Chercher d'abord du temps CPU critique : PLE/hachage, construction du graphe, synchronisations ou petits produits, puis comparer code optimisé/intrinsics à l'assembleur. CPU et GPU partagent la mémoire : une exécution simultanée n'ajoute pas mécaniquement de bande passante.

## Metal 4 sur notre M5

Apple décrit des types quantifiés supplémentaires et l'entrée directe de tenseurs coopératifs dans les opérations matricielles sur macOS 27. Cela permet d'éviter certains allers-retours par la mémoire de threadgroup. Les exigences de format et d'alignement restent importantes. [Présentation Apple WWDC26](https://developer.apple.com/videos/play/wwdc2026/330/).

MLX 0.32.2 utilise **déjà** NAX pour plusieurs QMM et gather QMM. Dans `QuantizedMatmul::eval_gpu`, la frontière QMV/QMM dépend de K/N et du GPU : sur notre architecture g17s, beaucoup de formes de vérification T2–T4 restent en QMV. Le gather trié demande notamment `B/E >= 4` ; avec 40 sélections pour 512 experts à T4, trier les routes ne suffit pas à faire entrer cette branche. Cela explique une limite structurelle de notre prototype `sorted-moe` ; ce n'est pas une attribution complète de sa régression mesurée.

L'expérience pertinente est un QMM **spécialisé pour la vérification courte**, éventuellement avec padding T4 → M16, ou une tuile qui regroupe les positions utilisant le même expert. Il faut vérifier si le calcul et le padding économisent plus que les coûts ajoutés. Le QMV actuel évite de matérialiser les poids BF16 et a ses propres réductions : passer à un matmul tensoriel avec conversion BF16 peut changer les logits. L'exactitude n'est donc pas acquise par construction.

L'autre piste est de fusionner des **frontières producteur/consommateur** : norm/gate → projection GDN, gate/up → SwiGLU, down → réduction des dix experts → injection HC, ou sélection QSA → lecture KV. Cela diffère du précédent `fused-moe` rejeté : chaque frontière doit avoir son propre prototype et sa propre attribution. Conserver gamma+1, casts BF16 et ordre des réductions est obligatoire. Le profil synchronisé met GDN/MoE en avant, mais ne fournit pas les pourcentages de temps GPU en exécution naturelle.

## Ce que font réellement oMLX et MTPLX

Révisions vérifiées sur les remotes : MTPLX `9882703f3105363ddc37eca9f97aa09a1d387112` ; oMLX `7f3a87f0afab070a3f4bee0558e49c99ea208830`. Le checkout oMLX antérieur `a435a373...` est conservé ; la nouvelle révision a été récupérée et inspectée avec `git show`, sans changer ce checkout.

| Projet et sources | Technique | Applicabilité à notre moteur |
|---|---|---|
| [oMLX oq.py](https://github.com/jundot/omlx/blob/7f3a87f0afab070a3f4bee0558e49c99ea208830/omlx/oq.py) | Budget de bits, protections par module, calibration imatrix oQe ; PLE Qwen4/g32 et protections MTP | Notre checkpoint bénéficie déjà de cette famille de recettes. Une nouvelle recette exige un nouveau contrôle de qualité |
| [oMLX m5_gather_qmm_nax.py](https://github.com/jundot/omlx/blob/7f3a87f0afab070a3f4bee0558e49c99ea208830/omlx/patches/m5_gather_qmm_nax.py) | Tuiles par expert, double buffering, SwiGLU dans l'épilogue, accès aux lignes source via un mapping | Surtout prefill avec assez de lignes par expert ; ne pas transposer les gains annoncés à nos quatre positions |
| [oMLX oq_a8_decode.h](https://github.com/jundot/omlx/blob/7f3a87f0afab070a3f4bee0558e49c99ea208830/omlx/custom_kernels/qwen35_prefill/csrc/oq_a8_decode.h) et [kernel INT8 NAX](https://github.com/jundot/omlx/blob/7f3a87f0afab070a3f4bee0558e49c99ea208830/omlx/custom_kernels/qwen35_prefill/csrc/qwen35_oq_a8_nax.metal) | Dépacking Q4/Q5 dans les registres, activations INT8, correction affine par groupe | Inspiration bas niveau ; approximation des activations, principalement prefill, opt-in avec seuil de 128 tokens dans son dispatcher |
| [MTPLX nax_verify.py](https://github.com/youssofal/MTPLX/blob/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/nax_verify.py) | Vérification Q4 M16 avec TensorOps, padding des petits blocs ; variantes SIMD split-K M4/M6 | Correspond directement à la géométrie MTP ; source attribuée à dflash-mlx. À comparer au QMV actuel, pas à un MLX non optimisé |
| [MTPLX qwen4_m4_stage3.py](https://github.com/youssofal/MTPLX/blob/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/qwen4_m4_stage3.py) | Routeur, gate/up, réduction down et résiduel spécialisés pour quatre positions | Certaines variantes attendent un pack affine Q4/g32 gate/up fusionné ; nos banques sont Q4/g64 et ne satisfont pas ce contrat |
| [MTPLX draft_lm_head.py](https://github.com/youssofal/MTPLX/blob/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/draft_lm_head.py) et [frspec_draft.py](https://github.com/youssofal/MTPLX/blob/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/frspec_draft.py) | Requantification de tête réservée au draft par morceaux, vocabulaire de fréquence 64K, cible complète | Piste directement réutilisable conceptuellement ; nos probes confirment l'économie locale, pas le gain total annoncé ailleurs |
| [MTPLX proj_quant.py](https://github.com/youssofal/MTPLX/blob/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/proj_quant.py) | Requantification optionnelle des projections résidentes Q8 → Q4, double quantification explicitée | Confirme l'intérêt d'auditer les poids toujours lus ; leurs résultats sur Hy3 ne s'appliquent pas à notre Qwen4 |
| [MTPLX kv_quant.py](https://github.com/youssofal/MTPLX/blob/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/kv_quant.py) | Stockage KV quantifié, avec chemin d'attention dédié ou déquantification de secours | Mémoire/contexte long ; ne garantit pas les sorties exactes du cache BF16 cible |

Les deux projets ont aussi batching, cache de prompt et stockage SSD. Ces fonctionnalités améliorent certaines requêtes ou la capacité serveur ; elles ne mesurent pas la vitesse brute d'une conversation avec cache neuf. Nos anciens adapters de recherche échouent la parité du prefill de ce checkpoint : aucune comparaison de vitesse stock oMLX/MTPLX n'est établie ici. Apache-2.0 et notices des sources tierces doivent être conservées si du code est repris.

Une mise à jour mérite une expérience séparée : [MLX 0.32.3](https://github.com/ml-explore/mlx/releases/tag/v0.32.3) corrige plusieurs chemins gather/SDPA et change notamment sigmoid et la propagation NaN des arg-réductions. Cela impose une nouvelle validation des oracles, kernels et caches ; ce n'est pas une mise à jour transparente ni une promesse de gain. Les guards de convolution et les blocs QMV bornés restent nécessaires jusqu'à une validation réelle. oMLX documente encore un garde sur les queues K de son gather NAX ; nos dimensions principales 640/2560/6144 sont multiples de 64, mais les autres formes doivent être vérifiées.

## Ordre des expériences proposées

1. **Tête du brouillon : Q6/Q4 et vocabulaire fréquent multilingue 64K.** Peu de modifications du modèle cible, preuve locale déjà disponible. Mesurer les trajectoires du brouillon et l'acceptation, puis quatre paires Rust alternées de 256 tokens et les quatre workloads chat.
2. **Vérification MTP M4 spécialisée Metal 4.** Comparer native QMV, notre QMV partagé et une tuile TensorOps, avec les poids BF16/mixed-bit réels. Tester logits, tous les états, rollback et validation GPU, avant le débit global.
3. **Fusions exactes limitées aux frontières GDN/MoE/HC.** Commencer par une frontière attribuée ; préserver les casts et les réductions. Le routeur doit produire exactement les mêmes experts, surtout près des égalités.
4. **Nouveau checkpoint de quantification résidente calibrée.** Variante clairement distincte avec corpus, qualité et performance. Le gain potentiel vient autant des résidents toujours lus que des banques MoE.
5. **Cache quantifié et attention directe par pages pour contexte long.** Comparer à contexte identique ; une attention lisant les pages directement est une autre expérience que le cache par blocs déjà rejeté.

Le réglage des intrinsics, layouts et registres s'applique surtout aux étapes 2–3. L'assembleur CPU n'est prioritaire qu'après preuve d'un coût CPU critique.

Passer de 68,43 à 100 tok/s demande **31,6 % de réduction du temps par token**, soit 46,1 % de débit supplémentaire. Le draft seul, environ 10 % du temps dans le cohort historique, ne peut pas suffire même s'il devenait gratuit. Il faut aussi gagner sur le vérificateur. Aucun résultat de cette étude n'atteint ou ne prédit avec confiance 100 tok/s.

## Reproduction et preuves

```sh
.venv/bin/python scripts/study_quantization.py \
  --model /Users/d9beud/.lmstudio/models/d9beuD/Qwen3.8-Flash-Next-oQ4e-mtp \
  --output results/quantization-study.json --probe
.venv/bin/python scripts/study_metal_toolchain.py \
  --output results/metal-toolchain-study.json
```

[Inventaire, échantillons et erreurs](../results/quantization-study.json), [tests du compilateur et capacités CPU](../results/metal-toolchain-study.json). Les rapports contiennent les empreintes des scripts/oracle/en-têtes, environnement, IDs et limitations. Aucun résultat instrumenté ou de composant n'a été ajouté à `performance-summary.json`. Les 25 tests release et validations Metal précédemment qualifiés concernent le moteur inchangé ; cette étude ne prétend pas avoir exécuté de nouveaux kernels de production.
