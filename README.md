<h1>Kickstarter Project Success Predictor</h1>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Kickstarter_logo.svg/1024px-Kickstarter_logo.svg.png" alt="Kickstarter logo" width="600" height="75">
<p>This repository contains a machine learning initiative designed to enhance the decision-making process for both project creators and backers on Kickstarter, the world's largest crowdfunding platform.</p>
<h3>Project Overview</h3>
Kickstarter has revolutionized the way ideas get funded and become reality. However, the success of a project on Kickstarter can be influenced by a multitude of factors. Recognizing this, I have embarked on developing two types of machine learning models: a classification model to predict the success or failure of projects at launch, and a clustering model to group similar projects to gain further insights.

<h4>I. Classification Model</h4>
<p>The Gradient-boosting Classifier model I have developed is a sophisticated tool that harnesses both temporal and textual data to enhance the predictive accuracy for Kickstarter project outcomes. For Kickstarter, the model's insights are a treasure trove for strategic decision-making, offering data-driven guidance to optimize support for project creators and enhance platform features. Project owners, armed with this model, can fine-tune their campaigns for better alignment with successful funding parameters, informed by a deep dive into historical data patterns. For backers, this translates into a higher caliber of projects to support, with a ripple effect of improved transparency and engagement in project presentations, leading to more informed backing decisions. 
</p>

<p>In my optimized cohort, I augmented the dataset with pivotal features: <I>Seasonality</I> to capture launch timing within seasonal trends, <I>Weekend Flags</I> for understanding the impact of launching or ending projects on weekends, and <I>Word Density</I> in project titles and descriptions for assessing content effectiveness. I also integrated a <I>Country-Specific Indicator</I> for U.S. projects, <I>Hourly Patterns</I> to examine the influence of time-of-day on project engagement, and a <I>Holiday Proximity Indicator</I> to explore the effect of major holidays on project success. These enhancements aim to deepen our understanding of factors driving Kickstarter outcomes.</p>

<h4>II. Clustering Model</h4>
Our goal was meticulously designed to dissect and understand the intricate array of attributes that characterize the diverse spectrum of projects on Kickstarter. By analyzing successful and failed Kickstarter projects, I refined our clustering approach and improved the silhouette score from 0.06 to 0.24. I used K-means to identify five distinct clusters and applied PCA and t-SNE for visualization, integrating our findings back into the original dataset for deeper insight. 
For an in-depth look at the defining features and insights of each cluster, check out our detailed report: https://github.com/kojen-coder/kickstarter-ml/blob/main/Cluster%20Model%20Report.pdf

