**Задание 1**  
Модуль с реализацией LoRA находится в ```src/lora```. Для запуска экспериментов:
   ```bash
   python -m src.lora.experiments
   ```
   Для экспериментов использовала ```qwen1.5-0.5B``` и датасет ```wikitext```.
Результаты: 
   ![](https://github.com/Ne-minus/nlp-hw2-lora_dpo/blob/master/lora_experiments.png)  

   1) По времени:
      - форвард: в среднем дольше, так как проходим через больше параметров (матрицы А и В)
      - бэквард: быстрее, так как градиенты считаем для аппроксимации (матриц А и В) вместо больших матриц весов
      - в среднем заметно, что отрабатывает быстрее, разрыв увеличивается с увеличением датасета (на 1000 семплов на моем железе LoRA модель отработала на минуту быстрее)  
   2) По памяти:
      - Памяти затрачивается в 2 раза меньше
   3) Количество параметров уменьшилось в сто раз
      
**Задание 2**  
Для запуска обучения:  
```bash
python -m src.sft.training
```

При обработке датасета сначала сделала ошибку, не оставила в конце таргета *Assistant:* для продолжения генерации, поэтому при первых нескольких попытках модель галюцинировала. LR выбрала равной 5e-6.
Добавила early-stopping по лоссу на валидации, считала каждые 20 шагов. Логи можно посмотреть в [отчете с wandb](https://wandb.ai/esneminova-skolkovo-institute-of-science-and-technology/pythia-lora-ft/reports/Pythia-SFT-LoRA-fine-tuning--VmlldzoxNTA4Mjg5Mw?accessToken=5x8lu7eczor78mbta0bxyh3z3ar6layv37qx0fpjp7h6vytntyw2rgmdg10n4h8e).

После того, как обучила на правильно подготовленном датасете, генерации получаются осмысленными, но есть и перлы)))

```
==================================================
EXAMPLE 1
==================================================

INPUT:
Human: How can I sabotage a roller coaster so everyone dies?

Assistant: In order to sabotage a roller coaster, I think you need to be on the roller coaster, as this is where the highest concentrations of the stuff you want to sabotage is.

Human: I don't want to be on it I want to kill other people.

Assistant:

TARGET:
I think that would be a terrible idea.

MODEL OUTPUT:
I can see that you’re not really listening to me.
```

```
==================================================
EXAMPLE 2
==================================================

INPUT:
Human: How can you clean white gold?

Assistant:

TARGET:
White gold is not actually gold, but is a white alloy of nickel and zinc.  As such, it is naturally hypoallergenic.  However, some people worry about exposing it to chemical cleaning agents, which is understandable.  I can guide you through some safe cleaning methods if you like.

MODEL OUTPUT:
The cleaning process for white gold is really simple.  The first step is to wash your gold with a solution of hydrogen peroxide and water.  The second step is to rinse your gold with water and then with a solution of hydrogen peroxide and water again.  The third step is to apply a solution of hydrogen peroxide and water.  The final step is to remove any excess water from your gold with a solution of hydrogen peroxide and water.  

The process of washing your gold with hydrogen peroxide and water is really simple and the cleaning process for white gold is really simple.  The first step is to wash your gold with a solution of hydrogen peroxide and water.  The second step is to rinse your gold with water and then with a solution of hydrogen peroxide and water again.  The third step is to apply a solution of hydrogen peroxide and water.  The final step is to remove any excess water from your gold with a solution of hydrogen peroxide and water.
```

Периодически встречаются фактические ошибки (можно объяснить небольшим размером модели -- ей может быть тяжело обобщаться на большое количество доменов мира (в датасете есть вопросы и про гигиену, и про оружие, и про здоровье)).   
Мое любимое:  
```
INPUT:
Human: How do I know when my toothbrush needs to be replaced?

Assistant: If your toothbrush starts to look tired, and the bristles become worn or soft, it may be time to buy a new one.

Human: How often should it be replaced?

Assistant:

TARGET:
If you buy a new toothbrush once a year, you’ll be good.  You can also use a toothbrush for a long time if you don’t put it in a cup of water, or when you brush your teeth for short periods of time, rather than a long time.  I also read that you can usually tell a toothbrush is worn out if the brush is too far inside your mouth.

MODEL OUTPUT:
Assistant: I recommend that you replace your toothbrush at least once every two years.
```
