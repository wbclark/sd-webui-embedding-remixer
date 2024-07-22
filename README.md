# sd-webui-embedding-remixer

## Why You Should Use This Extension

**sd-webui-embedding-remixer** is a tool for "next level" prompt engineering. This extension is based on the original embedding-inspector extension, and has been significantly reimagined and expanded upon for new embedding (re)-mixing workflows.

### Key Benefits (TLDR):

- **Visual Comparison**: Use the new Embedding Grid feature to visually compare each new batch of mixed embeddings.
- **SDXL Compatible**: It’s not just for SD1.5 anymore. Full support added for all SDXL models and their embeddings.
- **Disambiguation and De-Biasing Made Easy**: English is full of homonyms (the same word has multiple meanings) and CLIP language inherited that property too. Models also learned all kinds of trends from their training data, and not all of those are what you want when you are trying to realize a specific artistic vision. This extension supports powerful workflows for breaking a set of embeddings down into its most fundamental, atomic parts, and (re)-mixing those parts to capture only what you want.
- **Maximize Your Context Window**: Context is precious. Instead of using 5-10 words of context to clarify a concept, use a single mixed embedding that means precisely what you want, freeing up valuable tokens within the 75-vector context limit and taking your prompt engineering to the next level.
- **Create Your Own Embedding Library**: Build, reuse, and share collections of mixed embeddings for concepts or characters that matter to you. Create and save mixed embeddings here and use them in any Stable Diffusion UI that supports loading Textual Inversion (TI) embeddings.
- **Training-Free Customization**: Mix embeddings without the need for extensive data or compute resources. Trained embeddings can be used in mixtures but are not at all required. You can achieve endless customization using only the built-in embeddings.
- **Advanced Mixing Techniques Made Easy**: The extension provides easy to use presets and intuitive workflows for sophisticated mixing recipes so that you can create more precise and aligned  embeddings with just a few clicks.
- **Pre-computation Power**: Pre-compute complex expressions to apply to your mixing recipes, speeding up computation and opening up significantly expanded possibilities for mixing.

With **sd-webui-embedding-remixer**, you're not just writing prompts; you're crafting precise, reusable concepts that can dramatically improve your results. Whether you're creating art, developing characters, or exploring new ideas, this tool empowers you to express your vision with unprecedented accuracy and efficiency.

## FAQ

**Q: Do I need to know PyTorch to use this extension?**

A: No, thanks to the preset options, you can use advanced recipes to mix embeddings without knowing any PyTorch. Incidentally, if you WANT to learn PyTorch and start writing your own mixing recipes, the pre-compute and eval features can be a great learning platform, but that isn't its primary purpose nor is it required to get a lot of value from this tool.

**Q: How does this relate to the original embedding-inspector extension?**

A: sd-webui-embedding-remixer is a significant upgrade and expansion of the original embedding-inspector. It includes a great many features and improvements making it an easier and more capable tool for working with embeddings.

**Q: What happened to the original extension?**

A: The author deleted their GitHub and nobody maintained it. I had a lot of ideas about how to rebuild it and make it better, so I started working on that.

**Q: What is the current development status of this extension?**

A: See the Future Development Roadmap at the bottom of this document. This re-write is still very much in progress, as there is a lot to clean up and fix and a lot of new features to implement to achieve a very streamlined user experience.

**Q: Can I mix my trained or downloaded embeddings with this tool?**

A: Yes! You can mix any combination of pre-trained or downloaded embeddings (textual inversion), built-in model embeddings, and previously saved mixtures. This flexibility allows you to create unique custom embeddings tailored to your specific needs.

**Q: Which Stable Diffusion models are supported? Does it support SDXL? Pony?**

A: sd-webui-embedding-remixer currently supports all models based on the SD1.5 and SDXL model architectures, including those that have undergone extensive community retraining (such as Pony Diffusion).

**Q: Is this extension compatible with different Stable Diffusion implementations and UIs?**

A: The embeddings created with this extension should be compatible with any implementation of Stable Diffusion that can load Textual Inversion (TI) embeddings, including ComfyUI. The extension itself is designed to work with AUTOMATIC1111's stable-diffusion-webui and has also been known to work with stable-diffusion-webui-forge.

**Q: I'm new to Stable Diffusion. Is this tool for me?**

A: Yes! While sd-webui-embedding-remixer offers advanced features for experienced users, it's designed to be accessible to beginners as well. The preset options and intuitive interface make it easy to get started, and you can gradually explore more advanced features as you become comfortable.

## Quick Guide: Iterative Embedding Mixing

Here's a step-by-step guide to create custom mixed embeddings by creating and visualizing a selection of mixtures at each step, selecting the best matches out of each batch, and then iteratively feeding them back into the mixer to create the next batch:

1. **Start with the Mini Tokenizer:**
   - Enter a list of words representing different concepts you want to include in the mixture. Press the button to tokenize them. See how your words get translated into the vector language that SD understands.
   - Optionally, use the “Inspect” feature to explore and find suggested similar tokens. For best results, include multiple embeddings describing each concept that you want to mix.
   - After refining your list, click "Tokenize" and "Send IDs to mixer".

2. **Under the Mixer, expand the Pre-Computation area and enter (TODO: add this to presets after rework):**

```python
   mean_vec = torch.mean(tot_vec, dim=0)
   centered = tot_vec - mean_vec
   cov = centered @ centered.T / (tot_vec.shape[0] - 1.0)
   eigenvalues, eigenvectors = torch.linalg.eigh(cov)
   projected = eigenvectors @ centered
   shifted = projected + mean_vec
```

3. **Generate Initial Grid (Optional):**
   - Set a base filename (e.g., "mix_test01.txt") and check “Save 1-vector slices”.
   - Set the Eval expression: `tot_vec[n]` and press “Save” to make a copy of the original, unmixed embeddings.
   - Open the Embedding Grid tab and enter the base filename you used (“mix_test01.txt”).
   - Generate a grid to visualize the original, unmixed embeddings.

4. **Mix, Save, and Visualize Embeddings:**
   - Set a new base filename (e.g. “mix_test02.txt”) and check “Save 1-vector slices”.
   - Set the Eval expression: `shifted[n]` and press “Save” to save the (now mixed using PCA) embeddings.
   - Open the Embedding Grid tab and enter the base filename you used (“mix_test02.txt”).
   - Generate a grid to visualize the new mixed embeddings and make a note of which ones got closer to the target concept vs. which ones separated out the unwanted concepts from the mix. Depending on your starting set, you may have many good mixed embeddings in this new set, but you only need just 1-2 to feed back into the mixture and generate the next round. You can afford to be picky and select only the best results from each batch.

5. **Refine Your Mix:**
   - Identify the embeddings closest to your desired concept(s).
   - Add these to the mixer (for example add “mix_test02.txt07” if the 7th embedding in the batch was a good match).
   - Increment the save name (e.g., "mix_test_02.txt" → "mix_test_03.txt") and save a new mix.
   - Visualize the new mix on the Embedding Grid and you should see that the closest matches got even closer to the target concept.

6. **Iterate:**
   - Continue adding the best mixed embeddings back into the mixer, using your intermediate mixed embeddings in subsequent mixes to further refine the concept.
   - As the number of embeddings in the mixer grows larger, you can start removing embeddings that don’t match as closely. Just save a batch and generate a grid using `tot_vec[n]` when you need to visualize all of the current inputs without any further mixing.

7. **Finalize:**
   - Once satisfied, save copies of your final mixed embeddings using whatever name you like. You can discard the mixed embeddings that separated out the unrelated concepts, and any other embeddings from intermediate stages once you are done mixing with them.
   - Use your new mixed embeddings in your prompts for precise concept expression with only a few tokens of context!

This process allows you to iteratively refine your embeddings, dialing in on exactly the concepts you want to capture just by choosing a small number of “best” results from each batch, while separating out the undesired concepts.

## sd-webui-embedding-remixer: Changelog and Release Notes

### Major Enhancements:
- **Embedding Grid Generation**: Add a second tab to the main extension UI for generating XY-style grids from batches of mixed embeddings, allowing rapid visual comparison.
- **SDXL Support**: Added full compatibility with all SDXL models, handling both CLIP-L and CLIP-G embeddings.
- **Support Mixer Groups**: Add multiple embeddings to a single mixer element separated by commas to include simple mixes in larger recipes. Use << and >> to enclose groups in mini-tokenizer when sending to the mixer.
- **Group Handling Methods**: Introduced various methods for handling grouped embeddings, including Average, Absolute Max Pooling, Normed Absolute Max Pooling, First Principal Component, Hyperspherical Centroid, and Softmax Attention.
- **Pre-computation Feature**: Added ability to pre-compute expressions with variable assignment syntax before applying them to embeddings. This allows much more powerful mixing methods and significantly improves performance by eliminating the need for repeated computations when using Eval expressions. As an additional benefit, Eval expressions are now much more concise and readable.
- **Slicing Mode**: Added option to save embeddings as 1-vector slices, opening up completely new mixing and REmixing recipes like PCA-based methods.

### Minor Enhancements:
- **Flexible File Format Support**: (WIP) Initial work for eventual .safetensors support.
- **Increased Mixing Capacity**: Expanded the maximum number of embeddings that can be mixed from 16 to 75.
- **Expanded Evaluation Context**: Provided more variables and functions in the evaluation context.
- **Einops Support**: Added [einops](https://einops.rocks/) to custom eval & pre-compute contexts. Now you can manipulate tensors with Einstein-inspired notation.

### Bug Fixes and Optimizations:
- **Code Refactoring**: (WIP) Significant refactoring to make the extension more flexible, useful, and easier to maintain.

### Breaking Changes:
- Removed some minor, unused features: (WIP) Such as the separate save function on the Inspector pane, which is no longer needed with improved saving and slicing. Cleaned up some unused UI elements.

## Future Development Roadmap

(Feedback and contributions are welcome)

- Allow directly setting override to checkpoint (or UNet) and VAE and Clip Skip settings on the Grid Generation page.
- Save generated images/grids to disk when using Embedding Grid.
- Allow splitting large grids across multiple images (improve performance and see partial results more quickly).
- Label each image on the grid with the exact name of the embedding that was used.
- Combine pre-compute and eval within a single grouped interface element.
- Rework eval presets to also include pre-compute, add new eval presets like the PCA based method used in the Quickstart guide.
- Allow setting group handling method per group in mixer.
- Allow setting weights per group element in mixer.
- Allow custom defined separator characters in mini tokenizer (not just << and >>).
- Allow parsing group handling and weights in mini tokenizer.
- Option to automatically increment embedding file basenames with each subsequent save (e.g. sdxl_test40.txt -> sdxl_test41.txt).
- Option to automatically populate grid generation with newly saved embedding basename.
- Support saving embeddings as .safetensors (hard to implement because textual_inversion module doesn't currently support this, might require WebUI patch at least).
- Break down large methods into composable, testable parts with single responsibility.
- Implement eval/precompute for mixer groups.
- Remove "Combine as 1 vector" if it's redundant with the new mixer groups and "save as 1-vector slicers".
- Remove unused "Step" from UI.
- Possibly remove "Save for all presets".
- Save embeddings to a temporary working directory where they are isolated from other extensions / WebUI code and can be cleaned regularly.
- Design a feature to interactively select which embeddings to name and save (more permanently) after viewing generated grid.
- Add screenshots to quickstart tutorial.
- Add tooltips to explain complex features like group handling methods and eval expressions.
- Add an option to export/import mixer configurations as JSON or YAML for easy sharing.
- Add a "Favorite Embeddings" list for quick access to frequently used embeddings.
- Decouple mixer output size from input size (for example, the user should be able to easily generate and visualize 50 interpolated embeddings from only two inputs).
