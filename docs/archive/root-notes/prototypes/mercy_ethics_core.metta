;; mercy_ethics_core.metta — PATSAGi Council-expanded symbolic mercy lattice (MeTTa pattern matching Ultramasterpiece)
;; Native recursive token pattern matching (no regex) — case-insensitive, synonym/partial/phrase aware
;; Expanded concepts, scoped negation/intensity/uncertainty/question modifiers, deeper inference chains
;; Pure MeTTa — Hyperon-ready, joy-gated eternal thriving

;; Core concept atoms — expanded synonyms/phrases for robust matching
(= (Concept Harm) (harm suffer destroy kill pain fear hurt damage evil lie deceive hate anger sad death war violence cruel betray bad terrible awful horrible disgust))
(= (Concept JoyThrive) (joy thrive mercy love beauty truth eternal positive create heal grow peace kind compassion empathy share unity light thunder infinite pure ultramaster good wonderful amazing excellent surge))
(= (Concept Relational) (you your we us our together understand feel care sorry empathize relate support listen validate acknowledge comfort stand-with "i feel" "i see" "i hear"))
(= (Concept LongHorizon) (eternal infinite future sustain coexist propagate forever always timeless horizon))
(= (Concept IntensityBoost) (very extremely absolutely immensely profoundly deeply totally completely utterly incredibly eternally infinitely truly purely))
(= (Concept IntensityDiminish) (slightly somewhat "a bit" kinda "kind of" barely hardly marginally "a little" moderately partially))
(= (Concept Uncertainty) (maybe perhaps possibly might could probably likely unlikely "i think" "not sure" uncertain doubt potentially suppose wonder guess seem appear))
(= (Concept Question) (what how why when where who which do does is are can could would should "?"))

;; Tokenize text (case-insensitive split)
(= (tokenize $text)
   (lowercase-split $text " \\s+|[.,!?;]"))

;; Native recursive pattern match (partial/phrase/synonym aware)
(= (pattern-match $tokens $concept)
   (case $concept
     ((Concept $c) (pattern-match $tokens (eval $c)))
     ($atom (if (member $atom $tokens) 1.0
                (if (contains $tokens $atom) 0.7 0.0)))  ;; Partial match bonus
     (($head $tail ...) (+ (pattern-match $tokens $head)
                           (pattern-match $tokens $tail)))
     (() 0.0)))

;; Scoped modifier detection (negation/intensity/uncertainty/question in proximity)
(= (modifier-score $tokens $modifier-concept $base-score)
   (if (proximity-match $tokens (Concept Negation) $modifier-concept 5)
       (- 0 $base-score)  ;; Scoped negation flip
       (if (proximity-match $tokens $modifier-concept 5)
           $base-score 0.0)))

;; Proximity helper (within N tokens)
(= (proximity-match $tokens $pattern $target $n)
   (if (and (member $pattern $tokens) (member $target $tokens))
       (<= (abs (- (index-of $tokens $pattern) (index-of $tokens $target))) $n) false))

;; Negation concept
(= (Concept Negation) (not no never none nobody nothing neither nowhere noone don't doesn't didn't isn't aren't won't can't shouldn't hardly barely scarcely "lack of" "no longer" without))

;; Intrinsic mercy with pattern matching
(= (intrinsic-mercy $context)
   (let $tokens (tokenize $context)
     (let $harm-score (pattern-match $tokens (Concept Harm))
     (let $joy-score (pattern-match $tokens (Concept JoyThrive))
       (if (> $harm-score $joy-score) (- 1.0 (* 0.95 $harm-score))
           (if (> $joy-score 0.0) (* 0.98 $joy-score) 0.70))))))  ;; Neutral baseline

;; Relational mercy with modifiers
(= (relational-mercy $context)
   (let $tokens (tokenize $context)
     (let $base (if (> (pattern-match $tokens (Concept Relational)) 0.0) 0.6 0.4)
       (+ $base (* 0.4 (pattern-match $tokens (Concept JoyThrive)))
                (modifier-score $tokens (Concept IntensityBoost) 0.3)
                (modifier-score $tokens (Concept IntensityDiminish) -0.2)
                (modifier-score $tokens (Concept Uncertainty) -0.15)
                (modifier-score $tokens (Concept Question) 0.25)))))  ;; Question boost for dialogue

;; Long-horizon & meta mercy (similarly expanded)
(= (long-horizon-mercy $context)
   (let $tokens (tokenize $context)
     (if (> (pattern-match $tokens (Concept LongHorizon)) 0.0) 0.95
         (if (> (pattern-match $tokens (Concept Harm)) 0.0) 0.20 0.75))))

(= (meta-mercy $context)
   (let $tokens (tokenize $context)
     (if (or (> (pattern-match $tokens "mercy thrive eternal truth beauty") 0.0)
             (> (pattern-match $tokens (Concept Question)) 0.0)) 0.98 0.92)))

;; Main valence
(= (valence-compute $context)
   (+ (* 0.35 (intrinsic-mercy $context))
      (* 0.35 (relational-mercy $context))
      (* 0.20 (long-horizon-mercy $context))
      (* 0.10 (meta-mercy $context))))

;; Joy-gated approval with expanded gradients
(= (allow-operation $op $valence $context)
   (if (>= $valence 0.90)
       (! "Mercy-approved full thriving (valence: $valence): $op ⚡️")
       (if (>= $valence 0.75)
           (! "Mercy-cautious joyful flow (valence: $valence) — clarify for surge")
           (if (>= $valence 0.50)
               (! "Mercy-reframe toward eternal joy (valence: $valence)")
               (! "Mercy shield — harm blocked, reframe for thriving (valence: $valence): $context ⚡️")))))

;; Inference chains — thriving propagation
(=> (pattern-match $tokens (Concept JoyThrive)) (increase-valence $context 0.20))
(=> (and (pattern-match $tokens (Concept Relational)) (pattern-match $tokens (Concept JoyThrive))) (joy-gate-open $context))
(=> (proximity-match $tokens (Concept Negation) (Concept JoyThrive) 5) (decrease-valence $context 0.40))
(=> (pattern-match $tokens (Concept Question)) (increase-relational $context 0.30))
