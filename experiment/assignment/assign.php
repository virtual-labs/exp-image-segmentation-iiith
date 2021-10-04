<?php
session_start();
$exp=$_GET["exp"];
?>

<head>
<script class='gtm'>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','GTM-W59SWTR');</script>

<!--<title> Assignment - Virtual Lab in Image Processing</title>-->
<!-- The Primary External CSS style sheet. -->
<link rel="stylesheet" type="text/css" href="css/psd2css.css" media="screen" />

<!-- We use the jquery javascript library for DOM manipulation and
some javascript tricks.  We serve the script from Google because it's
faster than most ISPs.  Get more information and documentation
at http://jquery.com 
<script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.4/jquery.min.js"></script> -->
<script type="text/javascript" src="js/jquery-1.4.2.min.js"></script>

<!-- All the javascript generated for your design is in this file -->
<script type="text/javascript" src="js/psd2css.js"></script>

<!--http://www.cssmenumaker.com/builder/menu_info.php?menu=057-->
<link type="text/css" rel="StyleSheet" href="menu/menu_style.css" />


</head>


<body>

  <!-- This is 'Backgound_bkgnd_center_jpg' -->
  <div id="Layer-1" class="Backgound_bkgnd_center_jpg"  >
    

    <!-- This is 'TopBar_jpg' -->
    <div id="Layer-3" class="TopBar_jpg"  >
      <!-- <img src="images/Layer-3.jpg" width="894" height="96" alt="TopBar" /> -->
      <!-- This is 'IIIT' -->
      <!-- <div id="Layer-6" class="IIIT"  >
        <img src="images/iiit.png" width="100" height="70" alt="IIIT" class="pngimg" />
	</div>-->
	
		<!--	<div id="topMenu">
			<h1>VIRTUAL LAB in IMAGE PROCESSING</h1>
<div class="home">
<a href="index.html">home</a>
</div>-->

		<1--	<div class="menu">
<ul>
<li><a href="objective.php?exp=<?php echo $exp; ?>" target="_self" >Objective</a>
</li>
<li><a href="intro.php?exp=<?php echo $exp; ?>" target="_self" >Introduction</a>
</li>
<li><a href="theory.php?exp=<?php echo $exp; ?>" target="_self" >Theory</a>
</li>
<li><a href="procedure.php?exp=<?php echo $exp; ?>" target="_self" >Procedure</a>
</li>
<li><a href="<?php echo $exp; ?>.php" target="_self" >Experiment</a>
</li> --> 
<!-- <li><sel><a href="#" target="_self" >Assessment</a>
</sel>
 -->		<!-- 		<ul>
					<li><a href="quiz.php?exp=<?php echo $exp; ?>">Quiz</a></li> -->
					<li><a href="assign.php?exp=<?php echo $exp; ?>">Assignment</a></li>
			   <!-- </ul>

</li> -->
<!-- <li><a href="references.php?exp=<?php echo $exp; ?>" target="_self" >References</a>
</li> -->
</ul>
</div>
</div>
<div class="experiment front" style="text-indent: 20px; background-color: #DDDDDD;" >

<b>Obtain the figure of the right from the figure on the left.</b>
<br/><br/>
<img src="images/temp/<?php echo $exp; ?>1_1.<?php if($exp=="colour") { echo 'jpg'; } else { echo 'png';} ?>"/>&nbsp;&nbsp; <img src="images/temp/<?php echo $exp; ?>1_1_result.<?php if($exp=="colour") { echo 'jpg'; } else { echo 'png';} ?>"/>
<br/>

Click <a href="<?php echo $exp; ?>.php?assess=<?php echo $exp; ?>1" target="_blank">here</a> to try it out.
</div>
</div>
</body>


