<?php
session_start();
$_SESSION["piping"]=1;
?>

<head>
<script class='gtm'>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','GTM-W59SWTR');</script>

<title>Virtual Lab in Image Processing</title>
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
      <img src="images/Layer-3.jpg" width="894" height="96" alt="TopBar" />
      <!-- This is 'IIIT' -->
      <div id="Layer-6" class="IIIT"  >
        <img src="images/iiit.png" width="100" height="70" alt="IIIT" class="pngimg" />
	</div>
	
			<div id="topMenu">
			<h1 style="text-align: center;">Virtual Lab for IMAGE PROCESSING</h1>

			<div class="menu">
</div>
</div>
<div class="experiment" style="text-indent: 20px; background-color: #FFFFFF;
color: black;" >
<p><b>List of experiments:</b></p>


<ol start="1" type="1">	
	<li><a href='diff.php' >Distance and Connectivitiy</a></li>
	<li><a href='objective.php?exp=arith' >Arithmetic Operations</a></li>
	<li><a href='objective.php?exp=affine' >Affine Transformations</a></li>
	<li><a href='objective.php?exp=point' >Point Transformations</a></li>
       <li><a href='objective.php?exp=histo' >Image Histograms</a></li>
	<li><a href='objective.php?exp=neigh' >Neighbourhood Operations</a></li>
       <li><a href='fourier.php' >Fourier Transformations</a></li>
	<li><a href='morph.php' >Morphological Operations</a></li>
	<li><a href='objective.php?exp=colour' >Colour Image Processing</a></li>
       <li><a href='segment.php' >Image Segmentation</a></li>
	<li><a href='piping.php' >Piping Experiment</a></li>
      
  </ol></div>
</body>

